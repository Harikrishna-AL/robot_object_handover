from pathlib import Path
exec(open(Path(__file__).with_name('hand_fk.py')).read().split("if __name__")[0])
bychild={j.find('child').get('link'):j for j in joints};chain=[];link='hand_palm_link'
while link!='world':
 j=bychild[link];chain.insert(0,j);link=j.find('parent').get('link')
def calc(q,all=False):
 T=np.eye(4);poses=[];k=0
 for j in chain:
  o=j.find('origin');rpy=[0,0,0] if o is None else list(map(float,o.get('rpy','0 0 0').split()));xyz=[0,0,0] if o is None else list(map(float,o.get('xyz','0 0 0').split()))
  m=np.eye(4);m[:3,:3]=rot([0,0,1],rpy[2])@rot([0,1,0],rpy[1])@rot([1,0,0],rpy[0]);m[:3,3]=xyz
  R=np.eye(4)
  if j.get('type')!='fixed':R[:3,:3]=rot(list(map(float,j.find('axis').get('xyz').split())),q[k]);k+=1
  T=T@m@R;poses.append(T.copy())
 return poses if all else T
def log(R):
 t=np.arccos(np.clip((np.trace(R)-1)/2,-1,1));v=np.array([R[2,1]-R[1,2],R[0,2]-R[2,0],R[1,0]-R[0,1]])/2
 return v if t<1e-5 else v*t/max(np.sin(t),1e-9)
lo=np.array([-math.pi,-2.20,-math.pi,-2.53,-math.pi,-2.05,-math.pi]);hi=-lo
q0=np.array([0,-math.pi/8,0,3*math.pi/4,0,-math.pi/6,math.pi/4]);rng=np.random.default_rng(9)
found=[]
for x in [.08,.14,.20]:
 for z in [.55,.65,.75,.85]:
  goal=np.eye(4);goal[:3,3]=[x,0,z]
  best=None
  for seed in range(3):
   q=q0.copy() if seed==0 else np.clip(q0+rng.normal(0,.8,7),lo,hi)
   for it in range(120):
    T=calc(q);err=np.r_[goal[:3,3]-T[:3,3],.3*log(goal[:3,:3]@T[:3,:3].T)]
    if np.linalg.norm(err)<1e-5:break
    J=np.zeros((6,7));eps=1e-5
    for k in range(7):
     q2=q.copy();q2[k]+=eps;T2=calc(q2)
     J[:,k]=np.r_[(T2[:3,3]-T[:3,3])/eps,.3*log(T2[:3,:3]@T[:3,:3].T)/eps]
    dq=J.T@np.linalg.solve(J@J.T+.015**2*np.eye(6),err)
    dq*=min(1,.25/max(np.linalg.norm(dq),1e-9));q=np.clip(q+dq,lo,hi)
   T=calc(q);e=np.linalg.norm(goal[:3,3]-T[:3,3]);angle=np.linalg.norm(log(goal[:3,:3]@T[:3,:3].T))*180/math.pi
   if e<.005 and angle<3:
    # Approximate clearance between corresponding arm links on the two sides.
    poses=calc(q,True);P=np.array([t[:3,3] for t in poses[2:-2]])
    A=P.copy();A[:,:2]*=-1;A[:,0]-=.65;B=P.copy();B[:,0]-=1.25
    clearance=np.linalg.norm(A[:,None]-B[None,:],axis=-1).min()
    score=np.linalg.norm(q-q0)
    row=dict(target=[x,0,z],q=q.tolist(),position_error=e,angle_error=angle,arm_link_origin_clearance=float(clearance),score=score)
    if best is None or score<best['score']:best=row
  if best:found.append(best);print(json.dumps(best),flush=True)
json.dump(found,open('/tmp/true_palm_candidates.json','w'),indent=2)
