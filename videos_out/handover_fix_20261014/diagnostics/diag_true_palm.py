"""Score actual palm normals (+X, confirmed by palm_aux contact plates)."""
import argparse,json,os,sys
from omni.isaac.lab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=32)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(sys.argv[1:]+['--experience','/isaac-sim/apps/omni.isaac.sim.python.kit']);a.headless=True
launcher=AppLauncher(a);app=launcher.app
import torch,gymnasium as gym
from omni.isaac.lab.utils.math import combine_frame_transforms,subtract_frame_transforms,quat_apply
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks
cfg=parse_env_cfg('Isaac-Bimanual-Bidirectional-v0',device=a.device,num_envs=32)
cfg.use_object_weld=False;cfg.use_reverse_curriculum=False
E=gym.make('Isaac-Bimanual-Bidirectional-v0',cfg=cfg);u=E.unwrapped;E.reset()
dev=u.device;n=u.num_envs;ids=torch.arange(n,device=dev);role=ids%2==1
cands=json.load(open('/diagnostics/true_palm_candidates.json'))
arts=[u.scene.articulations[k] for k in u.cfg.keys];obj=u.scene.rigid_objects['object'];dt=u.sim.get_physics_dt()
palms=[r.find_bodies('hand_palm_link')[0][0] for r in arts]
deltas=torch.tensor([[0,0,0],[.005,0,0],[-.005,0,0],[0,0,.005]],device=dev)
shift=deltas[(ids%8)//2]
qs=[r.data.joint_pos.clone() for r in arts]
for i,r in enumerate(arts):
 qs[i][:,u._robot_joints_idx[i]]=torch.tensor([cands[e//8]['q'] for e in range(n)],device=dev)
 qs[i][:,u._hand_joints_idx[i]]=torch.where((role if i else ~role)[:,None],u.closed_hand_joints,u.open_hand_joints.expand(n,-1))
 r.write_joint_state_to_sim(qs[i],torch.zeros_like(qs[i]));r.set_joint_position_target(qs[i])
def ee(i):return arts[i].data.body_state_w[:,u.ee_jacobi_idx[i]+1,:7].clone()
def giver(x,y):return torch.where(role[:,None],y,x)
def rel():
 ep=giver(ee(0),ee(1));op=obj.data.root_state_w[:,:7]
 return subtract_frame_transforms(ep[:,:3],ep[:,3:],op[:,:3],op[:,3:])[0]
ep=giver(ee(0),ee(1));op,oq=combine_frame_transforms(ep[:,:3],ep[:,3:],u.cfg.obj_pos_trans+shift,u.cfg.obj_quat_trans)
obj.write_root_pose_to_sim(torch.cat((op,oq),-1));obj.write_root_velocity_to_sim(torch.zeros((n,6),device=dev))
r0=rel().clone();maxd=torch.zeros(n,device=dev);minz=op[:,2].clone();armdrift=torch.zeros(n,device=dev);collision=torch.zeros(n,device=dev)
ps=[r.data.body_state_w[:,palms[i],:3].clone() for i,r in enumerate(arts)];angles=[]
for i,r in enumerate(arts):
 norm=quat_apply(r.data.body_state_w[:,palms[i],3:7],torch.tensor([1.,0,0],device=dev).expand(n,-1))
 v=torch.nn.functional.normalize(ps[1-i]-ps[i],dim=-1)
 angles.append(torch.rad2deg(torch.acos((norm*v).sum(-1).clamp(-1,1))))
angles=torch.stack(angles,-1);sep=torch.norm(ps[0]-ps[1],dim=-1)
for t in range(900):
 u.scene.write_data_to_sim();u.sim.step(render=False);u.scene.update(dt)
 maxd=torch.maximum(maxd,torch.norm(rel()-r0,dim=-1));minz=torch.minimum(minz,obj.data.root_state_w[:,2])
 for i,r in enumerate(arts):armdrift=torch.maximum(armdrift,(r.data.joint_pos[:,u._robot_joints_idx[i]]-qs[i][:,u._robot_joints_idx[i]]).abs().max(-1)[0])
 if t%30==0:
  u.filter_collisions();collision=torch.maximum(collision,u.force_norms[:,-1])
u.filter_collisions();f=[u.force_norms[:,s].sum(-1) for s in [u._ur5e_contact_slice,u._gen3_contact_slice]]
rows=[]
for e in range(n):
 row=dict(env=e,candidate=e//8,giver=int(role[e]),arm_q=[cands[e//8]['q']]*2,obj_pos_trans=(u.cfg.obj_pos_trans[e]+shift[e]).tolist(),true_palm_angles=angles[e].tolist(),palm_separation=float(sep[e]),object_z0=float(op[e,2]),min_object_z=float(minz[e]),object_dz=float(obj.data.root_state_w[e,2]-op[e,2]),max_object_relative_drift=float(maxd[e]),max_arm_joint_drift=float(armdrift[e]),max_arm_collision_force=float(collision[e]),giver_contact_force=float(f[int(role[e])][e]),receiver_contact_force=float(f[1-int(role[e])][e]))
 rows.append(row);print('TRUE_PALM',json.dumps(row),flush=True)
json.dump(rows,open('/results/true_palm_search.json','w'),indent=2)
print('TRUE_PALM_SEARCH_DONE',flush=True)
E.close();app.close()
