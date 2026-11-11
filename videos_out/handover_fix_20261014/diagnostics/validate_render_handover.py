"""Render the real reset/zero-action path in both giving directions. No training."""
import argparse,json,os,sys
from omni.isaac.lab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--num_envs',type=int,default=1);p.add_argument('--steps',type=int,default=300)
AppLauncher.add_app_launcher_args(p)
a=p.parse_args(sys.argv[1:]+['--experience','/isaac-sim/apps/omni.isaac.sim.python.kit']);a.headless=True;a.enable_cameras=True
launcher=AppLauncher(a);app=launcher.app
import torch,numpy as np,gymnasium as gym,imageio.v2 as imageio
from omni.isaac.lab_tasks.utils import parse_env_cfg
from omni.isaac.lab.utils.math import subtract_frame_transforms, quat_apply
import omni.isaac.lab_tasks
selection=json.load(open('/diagnostics/selected_pose.json'))
cfg=parse_env_cfg('Isaac-Bimanual-Bidirectional-v0',device=a.device,num_envs=1)
# Use the actual corrected configuration, without diagnostic pose overrides.
assert cfg.reset_arm_joint_pos is not None
cfg.use_object_weld=False;cfg.use_reverse_curriculum=False;cfg.randomize_giver=True
E=gym.make('Isaac-Bimanual-Bidirectional-v0',cfg=cfg,render_mode='rgb_array');u=E.unwrapped
arts=[u.scene.articulations[k] for k in u.cfg.keys];obj=u.scene.rigid_objects['object']
metrics=[]
def surface_angles():
    poses=[r.data.body_state_w[:,r.find_bodies('hand_palm_link')[0][0],:7] for r in arts]
    out=[]
    for i in range(2):
        normal=quat_apply(poses[i][:,3:],torch.tensor([[1.,0,0]],device=u.device))
        toward=torch.nn.functional.normalize(poses[1-i][:,:3]-poses[i][:,:3],dim=-1)
        out.append(float(torch.rad2deg(torch.acos((normal*toward).sum(-1).clamp(-1,1)))[0]))
    return out
def ee(i):return arts[i].data.body_state_w[:,u.ee_jacobi_idx[i]+1,:7].clone()
def relative(i):
    ep=ee(i);op=obj.data.root_state_w[:,:7]
    return subtract_frame_transforms(ep[:,:3],ep[:,3:],op[:,:3],op[:,3:])[0]
for role in [0,1]:
    qrefs=[]
    for repeat in range(5):
        for attempt in range(30):
            E.reset()
            if int(u.giver_is_gen3[0])==role:break
        assert int(u.giver_is_gen3[0])==role
        qrefs.append(torch.cat([r.data.joint_pos[:,u._robot_joints_idx[i]].clone() for i,r in enumerate(arts)],-1))
    reset_error=max(float((q-qrefs[0]).abs().max()) for q in qrefs)
    start_angles=surface_angles();worst_angles=start_angles.copy()
    start_rel=relative(role).clone();start_z=obj.data.root_state_w[:,2].clone()
    palms=[r.find_bodies('hand_palm_link')[0][0] for r in arts]
    center=torch.stack([r.data.body_state_w[0,palms[i],:3] for i,r in enumerate(arts)]).mean(0).cpu().numpy()
    maxdrift=0.;resets=0;contact=[];receiver_contact=[];jointdrift=0.;peak_stage=0
    receiver=1-role
    recv_tips=arts[receiver].data.body_state_w[:,u.finger_tips[receiver],:3].mean(1)
    initial_receiver_distance=float(torch.norm(recv_tips-obj.data.root_state_w[:,:3],dim=-1)[0])
    name=['left_slot_gives','right_slot_gives'][role]
    writer=imageio.get_writer('/results/'+name+'_setup.mp4',fps=1/u.step_dt,codec='libx264',quality=8,macro_block_size=1)
    for t in range(a.steps):
        obs,reward,terminated,truncated,info=E.step(torch.zeros((1,18),device=u.device))
        resets+=int((terminated|truncated).sum())
        maxdrift=max(maxdrift,float(torch.norm(relative(role)-start_rel,dim=-1).max()))
        current=torch.cat([r.data.joint_pos[:,u._robot_joints_idx[i]] for i,r in enumerate(arts)],-1)
        jointdrift=max(jointdrift,float((current-qrefs[-1]).abs().max()))
        contact.append(float(u.force_norms[:,[u._ur5e_contact_slice,u._gen3_contact_slice][role]].sum()))
        receiver_contact.append(float(u.force_norms[:,[u._ur5e_contact_slice,u._gen3_contact_slice][receiver]].sum()))
        peak_stage=max(peak_stage,int(u.stage[0]))
        if t%10==0:worst_angles=[max(x,y) for x,y in zip(worst_angles,surface_angles())]
        if t<80: eye=center+np.array([.65,1.5,.65]);target=center+np.array([0,0,-.18])
        elif t<190: eye=center+np.array([.30,.65,.28]);target=center
        else: eye=center+np.array([-.45,-.85,.45]);target=center
        u.sim.set_camera_view(eye=eye.tolist(),target=target.tolist())
        u.sim.render()
        frame=u.render()
        if t == 20 and not np.any(frame):
            raise RuntimeError('Renderer still returns blank RGB frames after warm-up')
        writer.append_data(frame)
        if t in [30,120,230]:imageio.imwrite('/results/'+name+'_'+str(t)+'.png',frame)
    writer.close()
    row=dict(giver=role,reset_repeat_max_joint_error=reset_error,steps=a.steps,seconds=a.steps*u.step_dt,automatic_resets=resets,max_object_relative_drift=maxdrift,object_dz=float((obj.data.root_state_w[:,2]-start_z)[0]),max_arm_joint_drift=jointdrift,mean_giver_contact_force=float(np.mean(contact)),video=name+'_setup.mp4',peak_stage=peak_stage,initial_receiver_tips_to_object=initial_receiver_distance,max_receiver_contact_force=max(receiver_contact),initial_true_palm_angles=start_angles,max_true_palm_angles=worst_angles)
    metrics.append(row);print('VALIDATION',json.dumps(row),flush=True)
json.dump(metrics,open('/results/setup_validation.json','w'),indent=2)
print('SETUP_RENDER_DONE',flush=True)
E.close();app.close()
