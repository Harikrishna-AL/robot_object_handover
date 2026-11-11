import xml.etree.ElementTree as ET
import numpy as np,math,json
path='source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/manager_based/classic/aurova_reinforcement_learning/bimanual_handover/config/urdf/gen3_4f.urdf'
joints=ET.parse(path).getroot().findall('joint')
def rot(axis,t):
 a=np.array(axis);a=a/np.linalg.norm(a);x,y,z=a;c=np.cos(t);s=np.sin(t);v=1-c
 return np.array([[x*x*v+c,x*y*v-z*s,x*z*v+y*s],[y*x*v+z*s,y*y*v+c,y*z*v-x*s],[z*x*v-y*s,z*y*v+x*s,z*z*v+c]])
def fk(q,base):
 T={'world':base}
 todo=joints.copy()
 while todo:
  for j in todo.copy():
   parent=j.find('parent').get('link');child=j.find('child').get('link')
   if parent not in T:continue
   o=j.find('origin');rpy=[0,0,0] if o is None else list(map(float,o.get('rpy','0 0 0').split()));xyz=[0,0,0] if o is None else list(map(float,o.get('xyz','0 0 0').split()))
   m=np.eye(4);m[:3,:3]=rot([0,0,1],rpy[2])@rot([0,1,0],rpy[1])@rot([1,0,0],rpy[0]);m[:3,3]=xyz
   axis=j.find('axis');R=np.eye(4)
   if j.get('type')!='fixed':R[:3,:3]=rot(list(map(float,axis.get('xyz').split())),q.get(j.get('name'),0))
   T[child]=T[parent]@m@R;todo.remove(j)
 return T
if __name__=='__main__':
 rows=json.load(open('videos_out/handover_fix_20261014/pose_search.json'))
 for variant in ['baseline','default_joints']:
  row=next(r for r in rows if r['variant']==variant)
  out=[]
  for i in range(2):
   q=dict(zip([f'arm_joint_{j}' for j in range(1,8)],row['arm_q'][i]));b=np.eye(4);b[0,3]=[-.65,-1.25][i]
   if i==0:b[:3,:3]=rot([0,0,1],math.pi)
   T=fk(q,b)['hand_palm_link'];out.append(T)
  print(variant)
  for i in range(2):
   v=out[1-i][:3,3]-out[i][:3,3];v/=np.linalg.norm(v)
   print('arm',i,'palm',out[i][:3,3],'normal +X',out[i][:3,0],'toward-other angle',math.degrees(math.acos(np.clip(out[i][:3,0]@v,-1,1))))
