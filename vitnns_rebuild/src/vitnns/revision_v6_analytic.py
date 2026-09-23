"""Independent NumPy float64 Jacobians. No scanner, hooks, Torch, or autograd imports."""
import math
import numpy as np

def affine(value,jac,weight,bias):
    return value @ weight.T + bias, np.einsum('tij,oi->toj',jac,weight)

def layernorm(value,jac,weight,bias,eps):
    n=value.shape[-1]; centered=value-value.mean(-1,keepdims=True)
    variance=(centered*centered).mean(-1,keepdims=True); inv=1/np.sqrt(variance+eps)
    matrices=np.stack([np.diag(weight)*v @ (np.eye(n)-np.ones((n,n))/n-np.outer(c,c)/(n*(float(var)+eps))) for c,v,var in zip(centered,inv[:,0],variance[:,0])])
    return centered*inv*weight+bias,np.einsum('tij,tjk->tik',matrices,jac)

def gelu(value,jac):
    cdf=.5*(1+np.vectorize(math.erf)(value/np.sqrt(2)))
    derivative=cdf+value*np.exp(-value*value/2)/np.sqrt(2*np.pi)
    return value*cdf,derivative[:,:,None]*jac

def attention(value,jac,p,heads):
    z,jz=affine(value,jac,p['qkv.weight'],p['qkv.bias']);n,d=value.shape;dh=d//heads;N=jac.shape[-1]
    q,k,v=z.reshape(n,3,heads,dh).transpose(1,2,0,3)
    jq,jk,jv=jz.reshape(n,3,heads,dh,N).transpose(1,2,0,3,4)
    O=[];JQ=[];JK=[];JV=[]
    for h in range(heads):
        S=q[h]@k[h].T/np.sqrt(dh);e=np.exp(S-S.max(-1,keepdims=True));A=e/e.sum(-1,keepdims=True)
        dsq=np.einsum('tin,ui->tun',jq[h],k[h])/np.sqrt(dh)
        dsk=np.einsum('ti,uin->tun',q[h],jk[h])/np.sqrt(dh)
        soft=np.stack([np.diag(a)-np.outer(a,a) for a in A])
        daq=np.einsum('tuv,tvn->tun',soft,dsq);dak=np.einsum('tuv,tvn->tun',soft,dsk)
        O.append(A@v[h]);JQ.append(np.einsum('tun,ui->tin',daq,v[h]));JK.append(np.einsum('tun,ui->tin',dak,v[h]));JV.append(np.einsum('tu,uin->tin',A,jv[h]))
    cat=np.concatenate(O,axis=1);out=cat@p['proj.weight'].T+p['proj.bias']
    parts={}
    for key,js in [('query',JQ),('key',JK),('value',JV)]:parts[key]=np.einsum('tij,oi->toj',np.concatenate(js,axis=1),p['proj.weight'])
    for h in range(heads):
        j=np.zeros((n,d,N));j[:,h*dh:(h+1)*dh]=JQ[h]+JK[h]+JV[h]
        parts['head_'+str(h)]=np.einsum('tij,oi->toj',j,p['proj.weight'])
    return out,parts

def reference(x,params,heads=2,pre_ln=True,mlp=True):
    x=np.asarray(x,dtype=np.float64);n,d=x.shape;N=x.size;j=np.eye(N).reshape(n,d,N)
    u,ju=layernorm(x,j,params['norm1.weight'],params['norm1.bias'],1e-5) if pre_ln else (x,j)
    a,parts=attention(u,ju,{k[5:]:v for k,v in params.items() if k.startswith('attn.')},heads)
    r1=x+a; ja=parts['query']+parts['key']+parts['value'];jr1=j+ja
    if mlp:
        norm,jn=layernorm(r1,j,params['norm2.weight'],params['norm2.bias'],1e-5)
        f,jf=affine(norm,jn,params['mlp.fc1.weight'],params['mlp.fc1.bias']);f,jf=gelu(f,jf)
        b,jb=affine(f,jf,params['mlp.fc2.weight'],params['mlp.fc2.bias'])
    else:b=np.zeros_like(r1);jb=np.zeros_like(j)
    w=params['readout']; gy=(w.ravel()+jb.reshape(N,N).T@w.ravel()).reshape(n,d)
    maps={key:x.ravel()*np.einsum('ti,tin->n',gy,jp) for key,jp in parts.items()}
    maps['routing']=maps['query']+maps['key'];maps['identity']=x.ravel()*gy.ravel()
    maps['attention']=x.ravel()*np.einsum('ti,tin->n',gy,ja)
    maps['branch_reconstruction']=maps['query']+maps['key']+maps['value']
    maps['full']=x.ravel()*(jr1.reshape(N,N).T@gy.ravel())
    maps['with_bypass']=maps['identity']+maps['branch_reconstruction']
    return maps,float(np.sum((r1+b)*w))

def validate_use_maps(rows,expected,atol=1e-11,rtol=1e-9):
    """Reject missing, repeated, wrong-slot and merged producer uses even if totals close."""
    keys=[tuple(r['use']) for r in rows]
    if len(keys)!=len(set(keys)) or set(keys)!=set(expected):raise ValueError('USE_IDENTITY_MISMATCH')
    for row,key in zip(rows,keys):
        if not np.allclose(row['map'],expected[key],atol=atol,rtol=rtol):raise ValueError('USE_COMPONENT_MISMATCH')
    return True

def fanout_fixture():
    x=np.array([.7,-1.3,2.1]);W=np.array([[1.2,-.4,.8],[-.7,1.1,.3]])
    a=np.array([.9,-1.4]);b=np.array([-1.1,.6]);z=W@x
    # One producer enters two different inputs of f(z0,z1)=a.z0 + b.(z1*z1).
    maps={('producer','consumer',0,0):x*(W.T@a),('producer','consumer',1,0):x*(W.T@(2*b*z))}
    rows=[dict(use=list(k),map=v) for k,v in maps.items()]
    validate_use_maps(rows,maps)
    wrong=[dict(use=list(rows[0]['use']),map=rows[1]['map']),dict(use=list(rows[1]['use']),map=rows[0]['map'])]
    merged=[dict(use=['producer','consumer',-1,0],map=sum(maps.values()))]
    repeated=[dict(use=list(k),map=sum(maps.values())) for k in maps]
    results=[]
    for name,bad in [('wrong_slot',wrong),('merged_fanout',merged),('merged_fanout_duplicated',repeated)]:
        try:validate_use_maps(bad,maps)
        except ValueError as e:results.append(dict(mutation=name,rejected=True,reason=str(e),correct_pass=True))
        else:raise AssertionError('FALSE_PASS_'+name)
    return results,{str(k):v.tolist() for k,v in maps.items()}
