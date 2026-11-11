"""Physical pose/grip search. No policy, weld, rewards, or automatic resets."""
import argparse, json, os, sys
from omni.isaac.lab.app import AppLauncher
p=argparse.ArgumentParser()
p.add_argument('--num_envs',type=int,default=32)
p.add_argument('--out',default='/results')
p.add_argument('--steps',type=int,default=900)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(sys.argv[1:]+['--experience','/isaac-sim/apps/omni.isaac.sim.python.kit'])
a.headless=True
launcher=AppLauncher(a); app=launcher.app
import torch, gymnasium as gym
from omni.isaac.lab.utils.math import combine_frame_transforms, subtract_frame_transforms, euler_xyz_from_quat
from omni.isaac.lab_tasks.utils import parse_env_cfg
import omni.isaac.lab_tasks
sys.path.insert(0,'/workspace/isaaclab/source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/train')
from networks import SharedArmActorCriticPolicy
import numpy as np
policy=SharedArmActorCriticPolicy(gym.spaces.Box(-np.inf,np.inf,(50,),dtype=np.float32),gym.spaces.Box(-1,1,(18,),dtype=np.float32),lambda _:1e-4,arm_obs_dim=25,net_arch=[32,16])
optimized={id(p) for g in policy.optimizer.param_groups for p in g['params']}
print('POLICY_PARAMETERS_MISSING_FROM_OPTIMIZER',[name for name,p in policy.named_parameters() if id(p) not in optimized],flush=True)
del policy
cfg=parse_env_cfg('Isaac-Bimanual-Bidirectional-v0',device=a.device,num_envs=a.num_envs)
cfg.use_object_weld=False; cfg.use_reverse_curriculum=False
E=gym.make('Isaac-Bimanual-Bidirectional-v0',cfg=cfg); u=E.unwrapped
E.reset(); dev=u.device; n=u.num_envs; ids=torch.arange(n,device=dev)
arts=[u.scene.articulations[k] for k in u.cfg.keys]
obj=u.scene.rigid_objects['object']; dt=u.sim.get_physics_dt()
palms=[r.find_bodies('hand_palm_link')[0][0] for r in arts]
role=ids%2==1
rng=torch.Generator(device=dev).manual_seed(114)
offsets=torch.zeros((n//2,3),device=dev)
offsets[1:]=torch.randn((n//2-1,3),generator=rng,device=dev)*0.018
offsets=offsets.repeat_interleave(2,0)
base_ranges=u.ee_pose_ranges.clone()

def ee(i): return arts[i].data.body_state_w[:,u.ee_jacobi_idx[i]+1,:7].clone()
def give(x,y): return torch.where(role[:,None],y,x)
def relative():
    pose=give(ee(0),ee(1)); op=obj.data.root_state_w[:,:7]
    return subtract_frame_transforms(pose[:,:3],pose[:,3:],op[:,:3],op[:,3:])[0]
def step():
    u.scene.write_data_to_sim(); u.sim.step(render=False); u.scene.update(dt)
def place(qs):
    for i,r in enumerate(arts):
        hand=torch.where((role if i else ~role)[:,None],u.closed_hand_joints,u.open_hand_joints.expand(n,-1))
        qs[i][:,u._hand_joints_idx[i]]=hand
        r.write_joint_state_to_sim(qs[i],torch.zeros_like(qs[i])); r.set_joint_position_target(qs[i])
    ep=give(ee(0),ee(1))
    op,oq=combine_frame_transforms(ep[:,:3],ep[:,3:],u.cfg.obj_pos_trans+offsets,u.cfg.obj_quat_trans)
    obj.write_root_pose_to_sim(torch.cat((op,oq),-1)); obj.write_root_velocity_to_sim(torch.zeros((n,6),device=dev))
    return op.clone()

def geometry():
    ps=[r.data.body_state_w[:,palms[i],:3] for i,r in enumerate(arts)]
    fs=[r.data.body_state_w[:,u.finger_tips[i],:3].mean(1) for i,r in enumerate(arts)]
    angles=[]
    for i in range(2):
        d=torch.nn.functional.normalize(fs[i]-ps[i],dim=-1)
        v=torch.nn.functional.normalize(ps[1-i]-ps[i],dim=-1)
        angles.append(torch.rad2deg(torch.acos((d*v).sum(-1).clamp(-1,1))))
    return torch.stack(angles,-1),torch.norm(ps[0]-ps[1],dim=-1)

results=[]
for variant in ['baseline','facing_candidate','default_joints']:
    u.ee_pose_ranges.copy_(base_ranges)
    if variant=='facing_candidate':
        poses=torch.tensor([[.2994,.0797,.8606,-.4521,-.2032,.5853,-.6416],[.2902,.0297,.8606,-.4521,-.2032,.5853,-.6416]],device=dev)
        rr,pp,yy=euler_xyz_from_quat(poses[:,3:]); xyzrpy=torch.cat((poses[:,:3],torch.stack((rr,pp,yy),-1)),-1)
        u.ee_pose_ranges[:,:,0]=xyzrpy; u.ee_pose_ranges[:,:,1]=xyzrpy
    E.reset()
    qs=[r.data.joint_pos.clone() for r in arts]
    if variant=='default_joints': qs=[r.data.default_joint_pos.clone() for r in arts]
    start=place(qs); rel0=relative().clone(); angles,sep=geometry()
    maxd=torch.zeros(n,device=dev); minz=start[:,2].clone(); force_sum=torch.zeros(n,device=dev)
    for t in range(a.steps):
        step(); d=torch.norm(relative()-rel0,dim=-1); maxd=torch.maximum(maxd,d)
        minz=torch.minimum(minz,obj.data.root_state_w[:,2])
        if t>=a.steps-100:
            u.filter_collisions()
            f=[u.force_norms[:,s].sum(-1) for s in [u._ur5e_contact_slice,u._gen3_contact_slice]]
            force_sum+=torch.where(role,f[1],f[0])/100
    displacement=torch.norm(relative()-rel0,dim=-1)
    dz=obj.data.root_state_w[:,2]-start[:,2]
    for j in range(n):
        row=dict(variant=variant,env=j,giver=int(role[j]),offset=offsets[j].tolist(),angle=angles[j].tolist(),sep=sep[j].item(),z0=start[j,2].item(),min_z=minz[j].item(),dz=dz[j].item(),final_relative_drift=displacement[j].item(),max_relative_drift=maxd[j].item(),force=force_sum[j].item(),arm_q=[qs[i][j,u._robot_joints_idx[i]].tolist() for i in range(2)])
        results.append(row)
    print('VARIANT',variant,'best paired candidates:',flush=True)
    paired=[]
    for j in range(0,n,2):
        rows=results[-n+j:-n+j+2] if j+2<n else results[-2:]
        score=max(x['max_relative_drift'] for x in rows)
        paired.append((score,j))
    for score,j in sorted(paired)[:5]: print(json.dumps(results[-n+j]),json.dumps(results[-n+j+1]),flush=True)
os.makedirs(a.out,exist_ok=True)
with open(os.path.join(a.out,'pose_search.json'),'w') as f: json.dump(results,f,indent=2)
print('POSE_SEARCH_DONE',flush=True)
E.close();app.close()
