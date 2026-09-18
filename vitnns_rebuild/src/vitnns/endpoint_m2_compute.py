"""M2 all-head spatial replacement, preserving native prefix and suffix."""
from contextlib import contextmanager
import numpy as np
import torch
from .endpoint_nns import _frozen_eval
from .endpoint_vit import capture_unfused_attention_edges
from .endpoint_e4_intervention import intervention_scores, _check_model
from .endpoint_m2_regions import (BLOCKS, HEADS, REFERENCES, METHODS, COUNTS, GRIDS,
    slices, aggregate_map, region_orders, auc)

def close(a, b, label, atol=1e-4, rtol=1e-4):
    error = abs(float(a)-float(b))
    if error > atol + rtol*max(abs(float(a)), abs(float(b))):
        raise ValueError(f'{label}:{error}')
    return error

def clean_state(model, image, block, class_id):
    current = model.blocks[block]; observed = {}; hooks = []
    def capture_u(_m, args): observed['u'] = args[0]
    hooks.append(current.norm1.register_forward_pre_hook(capture_u))
    x = image.detach().clone().requires_grad_(True)
    try:
        with _frozen_eval(model), capture_unfused_attention_edges(current.attn, observed):
            logits = model(x); f = logits[0, class_id]
            edge = observed['projection_input']
            g = torch.autograd.grad(f, edge, retain_graph=True)[0]
            h = current.attn.num_heads; width = edge.shape[-1]//h
            maps = []; gradients = []
            for head in range(h):
                seed = torch.zeros_like(g)
                seed[:, :, head*width:(head+1)*width] = g[:, :, head*width:(head+1)*width]
                gx = torch.autograd.grad(edge, x, grad_outputs=seed.detach(), retain_graph=True)[0]
                maps.append((x.detach()*gx).detach().cpu().numpy())
                gradients.append(gx.detach().cpu().numpy())
            # Independent sum-of-head VJP identity. This is a branch cut, not the full logit cut.
            branch = x.detach()*torch.autograd.grad(edge, x, grad_outputs=g.detach())[0]
            reconstructed = np.stack(maps).astype(np.float64).sum(axis=0)
            actual = branch.detach().cpu().numpy().astype(np.float64)
            error = float(np.max(np.abs(actual-reconstructed)))
            if error > 1e-4 + 1e-4*float(np.max(np.abs(actual))): raise ValueError('HEAD_CLOSURE')
            return dict(score=float(f.detach()), predicted=int(logits.argmax()),
                u=observed['u'].detach(), edge=edge.detach(), gradient=g.detach(),
                attention=observed['attention_softmax'][0, :, 0, 1:].detach().cpu().numpy(),
                maps=maps, gradients=gradients, closure_error=error)
    finally:
        for hook in hooks: hook.remove()

def isolated_suffix(model, state, replacements, block, head, class_id, batch_size=64):
    """Only the selected clean head slice changes; all bypasses remain clean."""
    results = []; current = model.blocks[block]; h = current.attn.num_heads
    width = state['edge'].shape[-1]//h
    with _frozen_eval(model), torch.no_grad():
        for start in range(0, len(replacements), batch_size):
            part = replacements[start:start+batch_size]; n = len(part)
            edge = state['edge'].repeat(n, 1, 1).clone()
            edge[:, :, head*width:(head+1)*width] = part[:, :, head*width:(head+1)*width].to(edge)
            branch = current.attn.proj_drop(current.attn.proj(edge))
            tokens = state['u'].repeat(n, 1, 1) + current.drop_path1(current.ls1(branch))
            tokens = tokens + current.drop_path2(current.ls2(current.mlp(current.norm2(tokens))))
            for later in model.blocks[block+1:]: tokens = later(tokens)
            logits = model.forward_head(model.norm(tokens))
            if not bool(torch.isfinite(logits).all()): raise ValueError('ISOLATED_NONFINITE')
            results.extend(logits[:, class_id].double().cpu().tolist())
    return results

def perturbed(model, image, reference, selections, geometry, block, class_id, batch_size=32, progress=None):
    outputs = []; edges = []; regions = slices(geometry)
    for start in range(0, len(selections), batch_size):
        part = selections[start:start+batch_size]
        batch = image.repeat(len(part), 1, 1, 1)
        for i, selected in enumerate(part):
            if len(set(selected)) != len(selected): raise ValueError('DUP_REGION')
            for region in selected:
                ys, xs = regions[region]; batch[i, :, ys, xs] = reference[0, :, ys, xs]
        observed = {}
        def hook(_m, args): observed['edge'] = args[0].detach()
        handle = model.blocks[block].attn.proj.register_forward_pre_hook(hook)
        try:
            with _frozen_eval(model), torch.no_grad(): logits = model(batch)
        finally: handle.remove()
        if not bool(torch.isfinite(logits).all()) or not bool(torch.isfinite(observed['edge']).all()):
            raise ValueError('PERTURBED_NONFINITE')
        outputs.extend(logits[:, class_id].double().cpu().tolist()); edges.append(observed['edge'])
        if progress: progress()
    return outputs, torch.cat(edges)

def scan(model, image, references, task, progress=None):
    _check_model(model)
    block = task['block']; class_id = task['class_id']; model_id = task['model_id']
    if block not in BLOCKS or set(references) != set(REFERENCES): raise ValueError('SCOPE')
    if model.blocks[block].attn.num_heads != HEADS[model_id]: raise ValueError('HEAD_COUNT')
    state = clean_state(model, image, block, class_id); h = HEADS[model_id]
    if state['attention'].shape != (h,196): raise ValueError('ATTENTION_SHAPE')
    width = state['edge'].shape[-1]//h; rows=[]; checks=[]; score_tables=[]; candidate_tables=[]
    # Native-forward no-op verifies the optimized suffix for every image/block.
    noop = isolated_suffix(model, state, state['edge'], block, 0, class_id)[0]
    checks.append(dict(kind='noop', error=close(noop,state['score'],'NOOP'), expected=state['score'], observed=noop))
    for geometry in GRIDS:
        n = GRIDS[geometry]; count_list=COUNTS[geometry]
        fixed=[]
        for head in range(h):
            signed = aggregate_map(state['maps'][head], geometry, False)
            att = np.repeat(np.repeat(state['attention'][head].reshape(14,14),16,axis=0),16,axis=1)/256
            fixed.append(dict(endpoint_absolute=aggregate_map(state['maps'][head],geometry),
                endpoint_signed=[abs(v) for v in signed], signed_raw=signed,
                gradient_only=aggregate_map(state['gradients'][head],geometry),
                attention=[float(att[ys,xs].sum()) for ys,xs in slices(geometry)]))
        for ref_id in REFERENCES:
            full, edges = perturbed(model,image,references[ref_id],[[i] for i in range(n*n)],
                geometry,block,class_id,progress=progress)
            for head in range(h):
                lo,hi=head*width,(head+1)*width
                delta=(state['edge'][:,:,lo:hi]-edges[:,:,lo:hi]).double()
                g=state['gradient'][:,:,lo:hi].double()
                scores=dict(fixed[head], regional_taylor=(g*delta).sum((1,2)).abs().cpu().tolist(),
                    displacement_only=torch.linalg.vector_norm(delta,dim=(1,2)).cpu().tolist())
                key=f"{task['sample_id']}|{model_id}|{block}|{head}|{ref_id}"
                orders=region_orders(scores,geometry,key)
                score_tables.append(dict(geometry=geometry,reference=ref_id,head=head,scores=scores))
                selections=sorted({tuple(sorted(order['order'][:count]))
                    for order in orders for count in count_list if count>1})
                union_full,union_edges=perturbed(model,image,references[ref_id],selections,geometry,
                    block,class_id,progress=progress)
                single_isolated=isolated_suffix(model,state,edges,block,head,class_id)
                union_isolated=isolated_suffix(model,state,union_edges,block,head,class_id)
                cache={(i,):(full[i],single_isolated[i]) for i in range(n*n)}
                cache.update({s:(f,z) for s,f,z in zip(selections,union_full,union_isolated)})
                # One genuine regional replacement per head/reference/geometry checked against
                # the independent full clean-prefix intervention implementation.
                row=dict(component_id='parity',operation='head',block=block,head=head,num_heads=h)
                clean_native,native,_=intervention_scores(model,image,class_id=class_id,candidates=[row],
                    replacements={'parity':edges[0:1,:,lo:hi]})
                error=max(close(clean_native,state['score'],'CLEAN_NATIVE'),
                    close(native[0],single_isolated[0],'SUFFIX_NATIVE'))
                checks.append(dict(kind='suffix_native',geometry=geometry,reference=ref_id,head=head,
                    error=error,expected=native[0],observed=single_isolated[0]))
                candidates=[]
                for selection,(f,z) in sorted(cache.items()):
                    candidates.append(dict(selected=list(selection),full_logit=f,isolated_logit=z))
                candidate_tables.append(dict(geometry=geometry,reference=ref_id,head=head,candidates=candidates))
                best={count:{outcome:max(abs(state['score']-v[index]) for sel,v in cache.items() if len(sel)==count)
                    for outcome,index in [('full',0),('isolated',1)]} for count in count_list}
                for order in orders:
                    points=[]
                    for count in count_list:
                        selection=tuple(sorted(order['order'][:count])); f,z=cache[selection]
                        effects={'full':state['score']-f,'isolated':state['score']-z}
                        points.append(dict(count=count,area=count/(n*n),selected=list(selection),
                            signed=effects,absolute={k:abs(v) for k,v in effects.items()},
                            candidate_best=best[count],regret={k:best[count][k]-abs(v) for k,v in effects.items()}))
                    rows.append(dict(geometry=geometry,reference=ref_id,head=head,method=order['method'],
                        repeat=order['repeat'],seed=order['seed'],points=points,
                        auc={k:auc([p['absolute'][k] for p in points],count_list,geometry) for k in ('full','isolated')}))
                del union_edges,delta
                if progress: progress()
            del edges
    return dict(clean_score=state['score'],predicted=state['predicted'],closure_error=state['closure_error'],
        checks=checks,scores=score_tables,candidates=candidate_tables,curves=rows)
