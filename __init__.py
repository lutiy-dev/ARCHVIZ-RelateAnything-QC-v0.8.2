import json, math
import numpy as np
import torch
from PIL import Image, ImageDraw
try:
    from scipy.optimize import linear_sum_assignment
    HAS_SCIPY=True
except Exception:
    linear_sum_assignment=None; HAS_SCIPY=False
VERSION='0.9.0'

def _regions(v,name):
    if isinstance(v,list) and len(v)==1 and isinstance(v[0],dict): v=v[0]
    if not isinstance(v,dict): raise ValueError(f'{name} must be RA_REGIONS')
    b=np.asarray(v['boxes'],dtype=np.float32)
    if b.size==0: b=b.reshape(0,4)
    if b.ndim!=2 or b.shape[1]!=4: raise ValueError(f'{name}.boxes must be [N,4]')
    h,w=int(v['height']),int(v['width']); s=v.get('source_indices',list(range(len(b))))
    if len(s)!=len(b): s=list(range(len(b)))
    return b,h,w,[int(x) for x in s]

def _norm(b,h,w):
    if len(b)==0: return b.astype(np.float32),np.zeros((0,2),np.float32),np.zeros((0,2),np.float32)
    n=b.astype(np.float32).copy(); n[:,[0,2]]/=float(w); n[:,[1,3]]/=float(h)
    c=np.stack(((n[:,0]+n[:,2])*.5,(n[:,1]+n[:,3])*.5),1); s=np.stack((n[:,2]-n[:,0],n[:,3]-n[:,1]),1)
    return n,c,s

def _iou(a,b):
    x1=max(float(a[0]),float(b[0])); y1=max(float(a[1]),float(b[1])); x2=min(float(a[2]),float(b[2])); y2=min(float(a[3]),float(b[3]))
    iw=max(0.,x2-x1); ih=max(0.,y2-y1); inter=iw*ih
    aa=max(0.,float(a[2]-a[0]))*max(0.,float(a[3]-a[1])); bb=max(0.,float(b[2]-b[0]))*max(0.,float(b[3]-b[1])); u=aa+bb-inter
    return inter/u if u>0 else 0.

def _cluster(c,axis,tol):
    if len(c)==0:return np.zeros((0,),np.int32),[]
    order=np.argsort(c[:,axis]); groups=[]; cur=[int(order[0])]; mean=float(c[cur[0],axis])
    for x in order[1:]:
        x=int(x); v=float(c[x,axis])
        if abs(v-mean)<=tol: cur.append(x); mean=float(np.mean([c[i,axis] for i in cur]))
        else: groups.append(cur); cur=[x]; mean=v
    groups.append(cur); lab=np.full(len(c),-1,np.int32)
    for gi,g in enumerate(groups):
        for i in g: lab[i]=gi
    return lab,groups

def _neigh(c,k=4):
    out=np.zeros((len(c),k),np.float32)
    for i in range(len(c)):
        d=np.sort(np.linalg.norm(c-c[i],axis=1)); d=d[d>1e-8][:k]; vals=list(d)
        while len(vals)<k: vals.append(vals[-1] if vals else 0.)
        out[i]=vals
    return out

def _clamp(v): return min(1.,max(0.,float(v)))

def _cost(bn,bc,bs,br,bcol,bnei,an,ac,ass,ar,acol,anei,wc,wi,ws,wg,wn):
    weights=[max(0.,float(x)) for x in (wc,wi,ws,wg,wn)]; sw=sum(weights)
    if sw<=0: raise ValueError('At least one matching weight must be > 0')
    C=np.zeros((len(bn),len(an)),np.float32); detail={}; diag=math.sqrt(2.)
    brd=max(1,int(br.max()) if len(br) else 1); ard=max(1,int(ar.max()) if len(ar) else 1); bcd=max(1,int(bcol.max()) if len(bcol) else 1); acd=max(1,int(acol.max()) if len(acol) else 1)
    for i in range(len(bn)):
        for j in range(len(an)):
            center=_clamp(np.linalg.norm(bc[i]-ac[j])/diag); ioup=_clamp(1.-_iou(bn[i],an[j]))
            size=_clamp(np.mean(np.abs(np.log(np.maximum(ass[j],1e-6)/np.maximum(bs[i],1e-6)))))
            grid=_clamp(.5*abs(float(br[i])/brd-float(ar[j])/ard)+.5*abs(float(bcol[i])/bcd-float(acol[j])/acd))
            nei=_clamp(np.mean(np.abs(bnei[i]-anei[j])))
            val=(weights[0]*center+weights[1]*ioup+weights[2]*size+weights[3]*grid+weights[4]*nei)/sw
            C[i,j]=val; detail[(i,j)]={'center_penalty':center,'iou_penalty':ioup,'size_penalty':size,'grid_penalty':grid,'neighborhood_penalty':nei,'normalized_cost':float(val)}
    return C,detail

def _assign(C,dummy):
    n,m=C.shape
    if n==0 or m==0:return [],'rejection-aware/none'
    if HAS_SCIPY:
        big=1000.; M=np.full((n+m,n+m),big,np.float32); M[:n,:m]=C
        for i in range(n): M[i,m+i]=dummy
        for j in range(m): M[n+j,j]=dummy
        M[n:,m:]=0.; rr,cc=linear_sum_assignment(M)
        return [(int(r),int(c)) for r,c in zip(rr,cc) if r<n and c<m],'rejection-aware/hungarian-scipy'
    pairs=sorted((float(C[i,j]),i,j) for i in range(n) for j in range(m)); ub=set(); ua=set(); out=[]
    for cost,i,j in pairs:
        if i in ub or j in ua or cost>=2*dummy: continue
        ub.add(i); ua.add(j); out.append((i,j))
    return out,'rejection-aware/greedy-fallback'

def _structure(matches,bc,ac,row_tol,col_tol,order_tol):
    mp={m['before_idx']:m['after_idx'] for m in matches}; ids=sorted(mp); lr=[]; ud=[]; rows=[]; cols=[]; spacing=[]
    for p in range(len(ids)):
        for q in range(p+1,len(ids)):
            i,k=ids[p],ids[q]; j,l=mp[i],mp[k]
            bdx=float(bc[k,0]-bc[i,0]); adx=float(ac[l,0]-ac[j,0]); bdy=float(bc[k,1]-bc[i,1]); ady=float(ac[l,1]-ac[j,1])
            if abs(bdx)>order_tol and abs(adx)>order_tol and np.sign(bdx)!=np.sign(adx): lr.append([i,k])
            if abs(bdy)>order_tol and abs(ady)>order_tol and np.sign(bdy)!=np.sign(ady): ud.append([i,k])
            if (abs(bdy)<=row_tol)!=(abs(ady)<=row_tol): rows.append([i,k])
            if (abs(bdx)<=col_tol)!=(abs(adx)<=col_tol): cols.append([i,k])
            spacing.append(abs(float(np.linalg.norm(ac[l]-ac[j]))-float(np.linalg.norm(bc[k]-bc[i]))))
    return {'left_right_order_flips':lr,'above_below_order_flips':ud,'same_row_changes':rows,'same_column_changes':cols,'pair_spacing_drift_mean_norm':float(np.mean(spacing)) if spacing else 0.,'pair_spacing_drift_max_norm':float(np.max(spacing)) if spacing else 0.}

def _to_pil(t):
    if isinstance(t,list): t=t[0]
    a=np.clip(t[0].detach().cpu().float().numpy(),0,1); a=(a*255+.5).astype(np.uint8)
    if a.shape[-1]==4:a=a[...,:3]
    return Image.fromarray(a,'RGB')

def _fit(img,h=900):
    if img.height==h:return img
    return img.resize((max(1,int(round(img.width*h/img.height))),h),Image.Resampling.LANCZOS)

def _overlay(bim,aim,bb,ab,matches,missing,added,cwarn,swarn):
    b0,a0=_to_pil(bim),_to_pil(aim); b,a=_fit(b0),_fit(a0); gap=24; head=44
    cv=Image.new('RGB',(b.width+a.width+gap,b.height+head),(22,22,22)); cv.paste(b,(0,head)); cv.paste(a,(b.width+gap,head)); d=ImageDraw.Draw(cv)
    d.text((10,12),'BEFORE / GEOMETRY TRUTH',fill='white'); d.text((b.width+gap+10,12),'AFTER / AI RESULT',fill='white')
    bsx,bsy=b.width/b0.width,b.height/b0.height; asx,asy=a.width/a0.width,a.height/a0.height; green=(50,220,90); yellow=(255,210,60); red=(255,70,70); blue=(70,140,255)
    def box(rect,sx,sy,ox,col,label,w=3):
        x1,y1,x2,y2=map(float,rect); r=(x1*sx+ox,y1*sy+head,x2*sx+ox,y2*sy+head); d.rectangle(r,outline=col,width=w); d.text((r[0]+2,r[1]+2),label,fill=col); return r
    for m in matches:
        i,j=m['before_idx'],m['after_idx']; drift=m['center_drift_pct_diag']>cwarn or m['size_drift_pct_mean']>swarn; col=yellow if drift else green
        rb=box(bb[i],bsx,bsy,0,col,f'B{i}'); ra=box(ab[j],asx,asy,b.width+gap,col,f'A{j}'); d.line(((rb[0]+rb[2])*.5,(rb[1]+rb[3])*.5,(ra[0]+ra[2])*.5,(ra[1]+ra[3])*.5),fill=col,width=1)
    for i in missing: box(bb[i],bsx,bsy,0,red,f'MISS B{i}',4)
    for j in added: box(ab[j],asx,asy,b.width+gap,blue,f'ADD A{j}',4)
    return torch.from_numpy(np.asarray(cv,dtype=np.float32)/255.)[None,...]

def _stats(vals):
    if not vals:return {'min':None,'mean':None,'max':None}
    return {'min':float(np.min(vals)),'mean':float(np.mean(vals)),'max':float(np.max(vals))}

def _semantic(before_json,after_json):
    def parse(x):
        if x is None:return None
        if isinstance(x,list):x=x[0] if x else ''
        if not str(x).strip():return None
        r=json.loads(str(x)).get('relations'); return r if isinstance(r,list) else None
    try: br,ar=parse(before_json),parse(after_json)
    except Exception as e:return {'available':False,'reason':f'parse error: {e}'}
    if br is None or ar is None:return {'available':False,'reason':'relations_json not connected or empty'}
    sig=lambda r:(int(r.get('subject_source_idx',-1)),str(r.get('predicate','')),int(r.get('object_source_idx',-1)))
    bs={sig(r) for r in br}; aset={sig(r) for r in ar}; return {'available':True,'note':'Advisory only; never used as geometry truth.','removed':[list(x) for x in sorted(bs-aset)],'added':[list(x) for x in sorted(aset-bs)]}

class RACompareSmartQCV082:
    RETURN_TYPES=('IMAGE','STRING','STRING'); RETURN_NAMES=('qc_overlay','qc_report','qc_json'); FUNCTION='compare'; CATEGORY='RelateAnything/QC v0.9'; OUTPUT_NODE=True
    @classmethod
    def INPUT_TYPES(cls):
        F=lambda d,mi,ma,st:('FLOAT',{'default':d,'min':mi,'max':ma,'step':st})
        return {'required':{'before_image':('IMAGE',),'after_image':('IMAGE',),'before_regions':('RA_REGIONS',),'after_regions':('RA_REGIONS',),'row_tolerance':F(.02,.001,.2,.001),'column_tolerance':F(.02,.001,.2,.001),'weight_center':F(4.,0.,20.,.1),'weight_iou':F(1.5,0.,20.,.1),'weight_size':F(1.,0.,20.,.1),'weight_grid':F(1.5,0.,20.,.1),'weight_neighborhood':F(1.,0.,20.,.1),'accept_match_cost':F(.55,.01,1.,.01),'dummy_unmatched_cost':F(.35,.01,1.,.01),'warn_center_drift_pct':F(1.,0.,20.,.1),'warn_size_drift_pct':F(8.,0.,100.,.5),'fail_unmatched_fraction':F(.20,0.,1.,.01),'min_reliable_match_fraction':F(.50,0.,1.,.05),'order_tolerance':F(.01,.001,.2,.001)},'optional':{'before_relations_json':('STRING',{'forceInput':True}),'after_relations_json':('STRING',{'forceInput':True})}}
    def compare(self,before_image,after_image,before_regions,after_regions,row_tolerance,column_tolerance,weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood,accept_match_cost,dummy_unmatched_cost,warn_center_drift_pct,warn_size_drift_pct,fail_unmatched_fraction,min_reliable_match_fraction,order_tolerance,before_relations_json=None,after_relations_json=None):
        bb,bh,bw,bsrc=_regions(before_regions,'before_regions'); ab,ah,aw,asrc=_regions(after_regions,'after_regions'); bn,bc,bs=_norm(bb,bh,bw); an,ac,ass=_norm(ab,ah,aw)
        br,brg=_cluster(bc,1,row_tolerance); bcol,bcg=_cluster(bc,0,column_tolerance); ar,arg=_cluster(ac,1,row_tolerance); acol,acg=_cluster(ac,0,column_tolerance)
        C,det=_cost(bn,bc,bs,br,bcol,_neigh(bc),an,ac,ass,ar,acol,_neigh(ac),weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood); assigned,method=_assign(C,float(dummy_unmatched_cost))
        matches=[]; rejected=0; ub=set(); ua=set(); costs=[]; diag=math.sqrt(2.)
        for i,j in assigned:
            cost=float(C[i,j])
            if cost>accept_match_cost: rejected+=1; continue
            m=dict(det[(i,j)]); m.update(before_idx=i,after_idx=j,before_source_idx=bsrc[i],after_source_idx=asrc[j],before_row=int(br[i]),after_row=int(ar[j]),before_col=int(bcol[i]),after_col=int(acol[j]),center_drift_pct_diag=float(np.linalg.norm(ac[j]-bc[i])/diag*100.),size_drift_pct_mean=float(np.mean(np.abs(ass[j]-bs[i])/np.maximum(bs[i],1e-6))*100.)); matches.append(m); ub.add(i); ua.add(j); costs.append(cost)
        missing=[i for i in range(len(bb)) if i not in ub]; added=[j for j in range(len(ab)) if j not in ua]; unmatched=max(len(missing)/max(1,len(bb)),len(added)/max(1,len(ab))); reliable=len(matches)/max(1,max(len(bb),len(ab)))
        dstatus='FAIL' if unmatched>fail_unmatched_fraction else ('WARN' if missing or added else 'PASS'); st=_structure(matches,bc,ac,row_tolerance,column_tolerance,order_tolerance); cds=[m['center_drift_pct_diag'] for m in matches]; sds=[m['size_drift_pct_mean'] for m in matches]
        if reliable<min_reliable_match_fraction or len(matches)<2: gstatus='NOT_EVALUATED'
        else:
            hard=bool(st['left_right_order_flips'] or st['above_below_order_flips']); gw=bool((max(cds) if cds else 0)>warn_center_drift_pct or (max(sds) if sds else 0)>warn_size_drift_pct or st['same_row_changes'] or st['same_column_changes']); gstatus='FAIL' if hard else ('WARN' if gw else 'PASS')
        overall='FAIL' if dstatus=='FAIL' or gstatus=='FAIL' else ('WARN' if gstatus=='NOT_EVALUATED' or dstatus=='WARN' or gstatus=='WARN' else 'PASS'); sem=_semantic(before_relations_json,after_relations_json); ov=_overlay(before_image,after_image,bb,ab,matches,missing,added,warn_center_drift_pct,warn_size_drift_pct)
        diagstats={'candidate_cost':_stats(C.flatten().tolist() if C.size else []),'accepted_cost':_stats(costs),'assigned_real_pair_count':len(assigned),'rejected_pair_count':rejected,'reliable_match_fraction':reliable}
        res={'version':VERSION,'overall_status':overall,'detection_status':dstatus,'geometry_status':gstatus,'assignment_method':method,'scipy_available':HAS_SCIPY,'before':{'count':len(bb),'rows':len(brg),'columns':len(bcg)},'after':{'count':len(ab),'rows':len(arg),'columns':len(acg)},'matched_count':len(matches),'missing_before_indices':missing,'added_after_indices':added,'unmatched_fraction':unmatched,'reliable_match_fraction':reliable,'center_drift':{'mean_pct_diag':float(np.mean(cds)) if cds else 0.,'max_pct_diag':max(cds) if cds else 0.},'size_drift':{'mean_pct':float(np.mean(sds)) if sds else 0.,'max_pct':max(sds) if sds else 0.},'structure':st,'diagnostics':diagstats,'matches':matches,'semantic_relations':sem}
        fmt=lambda s:'n/a' if s['min'] is None else f"{s['min']:.3f} / {s['mean']:.3f} / {s['max']:.3f}"
        lines=[f'ARCHVIZ QC v{VERSION} — {overall}',f'Detection: {dstatus} | Geometry: {gstatus}',f'Assignment: {method}',f'Regions: BEFORE {len(bb)} → AFTER {len(ab)} | reliable matched {len(matches)}',f'Missing {len(missing)} | Added {len(added)} | unmatched {unmatched*100:.1f}%',f'Reliable match fraction: {reliable*100:.1f}%',f'Rows {len(brg)} → {len(arg)} | Columns {len(bcg)} → {len(acg)}',f'Candidate cost min/mean/max: {fmt(diagstats["candidate_cost"])}',f'Accepted cost min/mean/max: {fmt(diagstats["accepted_cost"])}',f'Assigned real pairs: {len(assigned)} | rejected by threshold: {rejected}',f'Center drift mean {res["center_drift"]["mean_pct_diag"]:.3f}% | max {res["center_drift"]["max_pct_diag"]:.3f}%',f'Size drift mean {res["size_drift"]["mean_pct"]:.2f}% | max {res["size_drift"]["max_pct"]:.2f}%',f'Order flips LR {len(st["left_right_order_flips"])} | UD {len(st["above_below_order_flips"])}',f'Grid changes rows {len(st["same_row_changes"])} | columns {len(st["same_column_changes"])}','Overlay: GREEN reliable | YELLOW drifted | RED missing | BLUE added','Geometry truth is deterministic; RelateAnything semantics are advisory only.']
        if gstatus=='NOT_EVALUATED': lines.append(f'Geometry NOT_EVALUATED: reliable match fraction {reliable:.3f} < required {min_reliable_match_fraction:.3f} or fewer than 2 reliable pairs.')
        report='\n'.join(lines); return {'ui':{'text':[report]},'result':(ov,report,json.dumps(res,ensure_ascii=False,indent=2))}



# v0.9.1 diagnostic full-set QC
VERSION_091 = '0.9.1'

def _ambiguity_info(C, i, j, threshold):
    alts=[]
    if C.shape[1] > 1:
        vals=[float(C[i,k]) for k in range(C.shape[1]) if k != j]
        if vals: alts.append(min(vals))
    if C.shape[0] > 1:
        vals=[float(C[k,j]) for k in range(C.shape[0]) if k != i]
        if vals: alts.append(min(vals))
    if not alts:
        return None, None, False
    alt=min(alts); margin=float(alt-float(C[i,j]))
    return alt, margin, bool(margin < float(threshold))

def _select_display_matches(matches, bc, count, mode, cwarn, swarn):
    if mode == 'all' or count <= 0 or len(matches) <= count:
        return list(matches)
    count=max(1,min(int(count),len(matches)))
    if mode == 'problems':
        def risk(m):
            cr=float(m.get('center_drift_pct_diag',0.0))/max(float(cwarn),1e-6)
            sr=float(m.get('size_drift_pct_mean',0.0))/max(float(swarn),1e-6)
            return (10.0 if m.get('ambiguous') else 0.0)+max(cr,sr)+float(m.get('normalized_cost',0.0))
        return sorted(matches,key=risk,reverse=True)[:count]
    ordered=sorted(matches,key=lambda m:(float(bc[m['before_idx'],1]),float(bc[m['before_idx'],0])))
    if count >= len(ordered): return ordered
    idx=np.linspace(0,len(ordered)-1,count).round().astype(int)
    return [ordered[int(i)] for i in idx]

def _overlay_v091(bim,aim,bb,ab,bsrc,asrc,matches,missing,added,cwarn,swarn,display_count,display_mode):
    b0,a0=_to_pil(bim),_to_pil(aim); b,a=_fit(b0),_fit(a0); gap=24; head=44
    cv=Image.new('RGB',(b.width+a.width+gap,b.height+head),(22,22,22)); cv.paste(b,(0,head)); cv.paste(a,(b.width+gap,head)); d=ImageDraw.Draw(cv)
    d.text((10,12),'BEFORE / GEOMETRY TRUTH',fill='white'); d.text((b.width+gap+10,12),'AFTER / AI RESULT',fill='white')
    bsx,bsy=b.width/b0.width,b.height/b0.height; asx,asy=a.width/a0.width,a.height/a0.height
    green=(50,220,90); yellow=(255,210,60); orange=(255,145,40); red=(255,70,70); blue=(70,140,255)
    def box(rect,sx,sy,ox,col,label,w=3):
        x1,y1,x2,y2=map(float,rect); r=(x1*sx+ox,y1*sy+head,x2*sx+ox,y2*sy+head)
        d.rectangle(r,outline=col,width=w); d.text((r[0]+2,r[1]+2),label,fill=col); return r
    display=_select_display_matches(matches,_norm(bb,b0.height,b0.width)[1],display_count,display_mode,cwarn,swarn)
    for m in display:
        i,j=m['before_idx'],m['after_idx']
        drift=m['center_drift_pct_diag']>cwarn or m['size_drift_pct_mean']>swarn
        col=orange if m.get('ambiguous') else (yellow if drift else green)
        rb=box(bb[i],bsx,bsy,0,col,f'B{int(bsrc[i]):02d}')
        ra=box(ab[j],asx,asy,b.width+gap,col,f'A{int(asrc[j]):02d}')
        d.line(((rb[0]+rb[2])*.5,(rb[1]+rb[3])*.5,(ra[0]+ra[2])*.5,(ra[1]+ra[3])*.5),fill=col,width=1)
    for i in missing: box(bb[i],bsx,bsy,0,red,f'UNMATCH B{int(bsrc[i]):02d}',4)
    for j in added: box(ab[j],asx,asy,b.width+gap,blue,f'UNMATCH A{int(asrc[j]):02d}',4)
    return torch.from_numpy(np.asarray(cv,dtype=np.float32)/255.)[None,...], len(display)

class RACompareDiagnosticQCV091:
    RETURN_TYPES=('IMAGE','STRING','STRING'); RETURN_NAMES=('qc_overlay','qc_report','qc_json'); FUNCTION='compare'; CATEGORY='RelateAnything/QC v0.9.1'; OUTPUT_NODE=True
    @classmethod
    def INPUT_TYPES(cls):
        F=lambda d,mi,ma,st:('FLOAT',{'default':d,'min':mi,'max':ma,'step':st})
        I=lambda d,mi,ma,st:('INT',{'default':d,'min':mi,'max':ma,'step':st})
        return {'required':{
            'before_image':('IMAGE',),'after_image':('IMAGE',),'before_regions':('RA_REGIONS',),'after_regions':('RA_REGIONS',),
            'row_tolerance':F(.02,.001,.2,.001),'column_tolerance':F(.02,.001,.2,.001),
            'weight_center':F(4.,0.,20.,.1),'weight_iou':F(1.5,0.,20.,.1),'weight_size':F(1.,0.,20.,.1),'weight_grid':F(1.5,0.,20.,.1),'weight_neighborhood':F(1.,0.,20.,.1),
            'accept_match_cost':F(.35,.01,1.,.01),'dummy_unmatched_cost':F(.17,.01,1.,.01),
            'ambiguity_margin_threshold':F(.03,0.,.2,.005),
            'warn_center_drift_pct':F(1.,0.,20.,.1),'warn_size_drift_pct':F(8.,0.,100.,.5),
            'fail_unmatched_fraction':F(.20,0.,1.,.01),'min_geometry_match_fraction':F(.50,0.,1.,.05),'order_tolerance':F(.01,.001,.2,.001),
            'detection_limit':I(32,1,512,1),'display_sample_count':I(8,0,128,1),'display_mode':(['problems','overview','all'],)
        },'optional':{'before_relations_json':('STRING',{'forceInput':True}),'after_relations_json':('STRING',{'forceInput':True})}}

    def compare(self,before_image,after_image,before_regions,after_regions,row_tolerance,column_tolerance,weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood,accept_match_cost,dummy_unmatched_cost,ambiguity_margin_threshold,warn_center_drift_pct,warn_size_drift_pct,fail_unmatched_fraction,min_geometry_match_fraction,order_tolerance,detection_limit,display_sample_count,display_mode,before_relations_json=None,after_relations_json=None):
        bb,bh,bw,bsrc=_regions(before_regions,'before_regions'); ab,ah,aw,asrc=_regions(after_regions,'after_regions')
        bn,bc,bs=_norm(bb,bh,bw); an,ac,ass=_norm(ab,ah,aw)
        br,brg=_cluster(bc,1,row_tolerance); bcol,bcg=_cluster(bc,0,column_tolerance); ar,arg=_cluster(ac,1,row_tolerance); acol,acg=_cluster(ac,0,column_tolerance)
        C,det=_cost(bn,bc,bs,br,bcol,_neigh(bc),an,ac,ass,ar,acol,_neigh(ac),weight_center,weight_iou,weight_size,weight_grid,weight_neighborhood)
        assigned,method=_assign(C,float(dummy_unmatched_cost))
        matches=[]; rejected=0; ub=set(); ua=set(); costs=[]; diag=math.sqrt(2.)
        for i,j in assigned:
            cost=float(C[i,j])
            if cost>accept_match_cost:
                rejected+=1; continue
            alt,margin,ambiguous=_ambiguity_info(C,i,j,ambiguity_margin_threshold)
            m=dict(det[(i,j)])
            m.update(before_idx=i,after_idx=j,before_source_idx=bsrc[i],after_source_idx=asrc[j],before_row=int(br[i]),after_row=int(ar[j]),before_col=int(bcol[i]),after_col=int(acol[j]),center_drift_pct_diag=float(np.linalg.norm(ac[j]-bc[i])/diag*100.),size_drift_pct_mean=float(np.mean(np.abs(ass[j]-bs[i])/np.maximum(bs[i],1e-6))*100.),alternative_cost=alt,ambiguity_margin=margin,ambiguous=ambiguous)
            matches.append(m); ub.add(i); ua.add(j); costs.append(cost)
        unmatched_b=[i for i in range(len(bb)) if i not in ub]; unmatched_a=[j for j in range(len(ab)) if j not in ua]
        denom=max(len(bb),len(ab)); accepted_fraction=(len(matches)/denom) if denom else None
        coverage_b=(len(matches)/len(bb)) if len(bb) else None; coverage_a=(len(matches)/len(ab)) if len(ab) else None
        ubf=(len(unmatched_b)/len(bb)) if len(bb) else None; uaf=(len(unmatched_a)/len(ab)) if len(ab) else None
        unmatched=max([x for x in (ubf,uaf) if x is not None],default=0.0)
        trunc_b=bool(detection_limit>0 and len(bb)>=int(detection_limit)); trunc_a=bool(detection_limit>0 and len(ab)>=int(detection_limit)); trunc=trunc_b or trunc_a
        insufficient=(len(bb)==0 or len(ab)==0)
        data_status='INSUFFICIENT_DATA' if insufficient else ('WARN' if trunc else 'PASS')
        ambiguous_matches=[m for m in matches if m.get('ambiguous')]; geom_matches=[m for m in matches if not m.get('ambiguous')]
        geom_fraction=(len(geom_matches)/denom) if denom else None
        if insufficient:
            matching_status='NOT_EVALUATED'
        elif unmatched>fail_unmatched_fraction:
            matching_status='FAIL'
        elif unmatched_b or unmatched_a or rejected or ambiguous_matches:
            matching_status='WARN'
        else:
            matching_status='PASS'
        st=_structure(geom_matches,bc,ac,row_tolerance,column_tolerance,order_tolerance)
        cds=[m['center_drift_pct_diag'] for m in geom_matches]; sds=[m['size_drift_pct_mean'] for m in geom_matches]
        if insufficient or geom_fraction is None or geom_fraction<min_geometry_match_fraction or len(geom_matches)<2:
            gstatus='NOT_EVALUATED'
        else:
            hard=bool(st['left_right_order_flips'] or st['above_below_order_flips'])
            gw=bool((max(cds) if cds else 0)>warn_center_drift_pct or (max(sds) if sds else 0)>warn_size_drift_pct or st['same_row_changes'] or st['same_column_changes'])
            gstatus='FAIL' if hard else ('WARN' if gw else 'PASS')
        if insufficient:
            overall='INSUFFICIENT_DATA'
        elif matching_status=='FAIL' or gstatus=='FAIL':
            overall='FAIL'
        elif data_status=='WARN' or matching_status=='WARN' or gstatus in ('WARN','NOT_EVALUATED'):
            overall='WARN'
        else:
            overall='PASS'
        sem=_semantic(before_relations_json,after_relations_json)
        ov,shown=_overlay_v091(before_image,after_image,bb,ab,bsrc,asrc,matches,unmatched_b,unmatched_a,warn_center_drift_pct,warn_size_drift_pct,display_sample_count,display_mode)
        diagstats={'candidate_cost':_stats(C.flatten().tolist() if C.size else []),'accepted_cost':_stats(costs),'assigned_real_pair_count':len(assigned),'rejected_pair_count':rejected,'accepted_match_fraction':accepted_fraction,'coverage_before':coverage_b,'coverage_after':coverage_a,'geometry_eligible_fraction':geom_fraction,'ambiguous_pair_count':len(ambiguous_matches)}
        res={'version':VERSION_091,'overall_status':overall,'data_sufficiency_status':data_status,'matching_status':matching_status,'geometry_status':gstatus,'assignment_method':method,'scipy_available':HAS_SCIPY,
             'full_regions_entering_qc':{'before':len(bb),'after':len(ab),'detection_limit':int(detection_limit),'possible_truncation_before':trunc_b,'possible_truncation_after':trunc_a},
             'matching':{'accepted_pairs':len(matches),'rejected_by_threshold':rejected,'ambiguous_pairs':len(ambiguous_matches),'unmatched_before_candidate_indices':unmatched_b,'unmatched_after_candidate_indices':unmatched_a,'unmatched_before_source_ids':[bsrc[i] for i in unmatched_b],'unmatched_after_source_ids':[asrc[j] for j in unmatched_a],'accepted_match_fraction':accepted_fraction,'coverage_before':coverage_b,'coverage_after':coverage_a},
             'geometry':{'eligible_pairs':len(geom_matches),'eligible_fraction':geom_fraction,'rows_before':len(brg),'rows_after':len(arg),'columns_before':len(bcg),'columns_after':len(acg),'center_drift':{'mean_pct_diag':float(np.mean(cds)) if cds else None,'max_pct_diag':max(cds) if cds else None},'size_drift':{'mean_pct':float(np.mean(sds)) if sds else None,'max_pct':max(sds) if sds else None},'structure':st},
             'diagnostics':diagstats,'matches':matches,'display':{'mode':display_mode,'shown_pairs':shown,'accepted_pairs':len(matches)},'semantic_relations':sem}
        fmt=lambda s:'n/a' if s['min'] is None else f"{s['min']:.3f} / {s['mean']:.3f} / {s['max']:.3f}"
        pct=lambda x:'N/A' if x is None else f'{x*100:.1f}%'
        val=lambda x,d=3:'n/a' if x is None else f'{x:.{d}f}'
        lines=[
            f'ARCHVIZ QC v{VERSION_091} — {overall}',
            f'Data sufficiency: {data_status} | Matching: {matching_status} | Geometry: {gstatus}',
            f'Assignment: {method}',
            f'FULL REGIONS ENTERING QC: BEFORE {len(bb)} → AFTER {len(ab)}',
            f'Possible truncation at limit {int(detection_limit)}: BEFORE {"YES" if trunc_b else "no"} | AFTER {"YES" if trunc_a else "no"}',
            f'FULL MATCHING: accepted {len(matches)} | rejected threshold {rejected} | ambiguous {len(ambiguous_matches)}',
            f'Unmatched BEFORE: {len(unmatched_b)} candidates | Unmatched AFTER: {len(unmatched_a)} candidates',
            f'Accepted match fraction: {pct(accepted_fraction)} | coverage BEFORE {pct(coverage_b)} | AFTER {pct(coverage_a)}',
            f'Geometry-eligible pairs: {len(geom_matches)} | fraction {pct(geom_fraction)}',
            f'Rows {len(brg)} → {len(arg)} | Columns {len(bcg)} → {len(acg)}',
            f'Candidate cost min/mean/max: {fmt(diagstats["candidate_cost"])}',
            f'Accepted cost min/mean/max: {fmt(diagstats["accepted_cost"])}',
            f'Center drift mean {val(float(np.mean(cds)) if cds else None)}% | max {val(max(cds) if cds else None)}%',
            f'Size drift mean {val(float(np.mean(sds)) if sds else None,2)}% | max {val(max(sds) if sds else None,2)}%',
            f'Order flips LR {len(st["left_right_order_flips"])} | UD {len(st["above_below_order_flips"])}',
            f'Grid changes rows {len(st["same_row_changes"])} | columns {len(st["same_column_changes"])}',
            'Overlay: GREEN stable | YELLOW drifted | ORANGE ambiguous | RED unmatched BEFORE candidate | BLUE unmatched AFTER candidate',
            'Unmatched regions are candidates only; detector benchmark with GT is required to confirm misses / false positives.'
        ]
        if insufficient: lines.append('Geometry NOT_EVALUATED: one or both full region sets are empty.')
        elif geom_fraction is not None and (geom_fraction<min_geometry_match_fraction or len(geom_matches)<2): lines.append(f'Geometry NOT_EVALUATED: eligible fraction {geom_fraction:.3f} < required {min_geometry_match_fraction:.3f} or fewer than 2 unambiguous pairs.')
        if trunc: lines.append('POSSIBLE_TRUNCATION: a full region count reached the configured detection limit.')
        report='\n'.join(lines)
        return {'ui':{'text':[report]},'result':(ov,report,json.dumps(res,ensure_ascii=False,indent=2))}

NODE_CLASS_MAPPINGS={'RACompareSmartQCV082':RACompareSmartQCV082,'RACompareSmartQCV09':RACompareSmartQCV082,'RACompareDiagnosticQCV091':RACompareDiagnosticQCV091}
NODE_DISPLAY_NAME_MAPPINGS={'RACompareSmartQCV082':'RA · SMART BEFORE vs AFTER QC · v0.8.2 (legacy)','RACompareSmartQCV09':'RA · SMART BEFORE vs AFTER QC · v0.9 (legacy full report after sampled input)','RACompareDiagnosticQCV091':'RA · DIAGNOSTIC FULL-SET QC · v0.9.1'}


# v0.9 stable paired sampling extension
from .stable_sampler_v09 import RAStablePairSamplerV09
NODE_CLASS_MAPPINGS["RAStablePairSamplerV09"] = RAStablePairSamplerV09
NODE_DISPLAY_NAME_MAPPINGS["RAStablePairSamplerV09"] = "RA · STABLE PAIRED REGION SAMPLER · v0.9"
