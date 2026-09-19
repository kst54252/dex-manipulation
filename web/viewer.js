// Three.js/OrbitControls are general rendering dependencies; see THREE_LICENSE.txt.
import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const data = JSON.parse(document.getElementById('pose-data').textContent);
const $ = id => document.getElementById(id);
const matrix = rows => new THREE.Matrix4().set(...rows.flat());
const unitScale = new THREE.Vector3(1, 1, 1);
const colors = [0xd97706, 0x0891b2, 0x7c3aed, 0xdb2777, 0x059669];
const fingers = ['thumb', 'index', 'middle', 'ring', 'little'];
const chains = fingers.map(f => [data.semantic_names.indexOf('wrist'), ...['mcp','pip','dip','tip'].map(j=>data.semantic_names.indexOf(`${f}_${j}`))]);
const jointColors = data.semantic_names.map(s => s === 'wrist' ? 0x334155 : colors[fingers.findIndex(f => s.startsWith(f+'_'))]);
const model = data.model;
const reference = data.reference;
const referenceImages = reference ? reference.images.map(src=>{const img=new Image();img.src=src;return img;}) : [];
const referenceLoaded = Promise.all(referenceImages.map(img=>img.decode()));
if(reference){$('reference-panel').hidden=false;$('reference-options').hidden=false;document.querySelector('.views').classList.add('with-reference');}
else{$('camera-mode').value='orbit';$('camera-mode').querySelector('[value=dataset]').disabled=true;}
const referenceCanvas=$('reference'),referenceContext=referenceCanvas.getContext('2d');
if(reference){referenceCanvas.width=reference.width;referenceCanvas.height=reference.height;}
function drawReference(index){
  if(!reference || !referenceImages[index].complete)return;
  referenceContext.drawImage(referenceImages[index],0,0,reference.width,reference.height);
  if($('rgb-points').checked){
    const points=reference.points_2d[index];
    chains.forEach((chain,f)=>{
      referenceContext.strokeStyle='#'+colors[f].toString(16).padStart(6,'0');referenceContext.lineWidth=2;
      referenceContext.beginPath();chain.forEach((k,j)=>{const p=points[k];if(j)referenceContext.lineTo(...p);else referenceContext.moveTo(...p);});referenceContext.stroke();
    });
    points.forEach((p,k)=>{referenceContext.fillStyle='#'+jointColors[k].toString(16).padStart(6,'0');referenceContext.beginPath();referenceContext.arc(...p,3,0,Math.PI*2);referenceContext.fill();});
  }
}
const joints = model.joints.map(j => ({...j, f0: matrix(j.frame0), f1inv: matrix(j.frame1).invert(), axisVector: new THREE.Vector3(...j.axis), index:model.full_names.indexOf(j.name)}));
const validMatrix = rows => rows && rows.flat().every(x => x !== null && Number.isFinite(x));
const decompose = rows => {
  if (!validMatrix(rows)) return null;
  const p = new THREE.Vector3(), q = new THREE.Quaternion(), s = new THREE.Vector3();
  matrix(rows).decompose(p,q,s);return {p,q};
};
const wristPoses = data.wrist.map(decompose), objectPoses = data.object_poses.map(decompose);
const blendPose = (a,b,alpha) => new THREE.Matrix4().compose(a.p.clone().lerp(b.p,alpha),a.q.clone().slerp(b.q,alpha),unitScale);
function fk(active,wrist){
  const full = model.coupling.map((row,i)=>row.reduce((sum,w,k)=>sum+w*active[k],model.offset[i]));
  const links = {[model.root]: wrist};
  for(const j of joints){
    let motion = new THREE.Matrix4();
    if(j.kind==='revolute') motion.makeRotationAxis(j.axisVector,full[j.index]);
    else if(j.kind==='prismatic') motion.makeTranslation(...j.axis.map(x=>x*full[j.index]));
    let local=j.f0.clone().multiply(motion).multiply(j.f1inv);
    if(j.reversed)local.invert();
    links[j.child]=links[j.parent].clone().multiply(local);
  }
  const points=model.keypoints.map(k=>new THREE.Vector3(...k.xyz).applyMatrix4(links[k.link]).toArray());
  return {links,points,full};
}
function surface(v,f){
  const geometry=new THREE.BufferGeometry();
  geometry.setAttribute('position',new THREE.Float32BufferAttribute(v.flat(),3));
  geometry.setIndex(f.flat());geometry.computeVertexNormals();return geometry;
}
const collisionGeometries=data.geometry.map(g=>surface(g.vertices,g.faces));
const canGeometry=data.object_mesh?surface(data.object_mesh.vertices,data.object_mesh.faces):null;
const sphereGeometry=new THREE.SphereGeometry(1,10,8), boneGeometry=new THREE.CylinderGeometry(1,1,1,7);
const bounds=new THREE.Box3(new THREE.Vector3(...data.bounds[0]),new THREE.Vector3(...data.bounds[1]));
const center=bounds.getCenter(new THREE.Vector3()), radius=bounds.getSize(new THREE.Vector3()).length()/2;
const camera=new THREE.PerspectiveCamera(43,1,0.001,20);camera.up.set(0,-1,0);
const resetCamera=()=>{camera.position.copy(center).add(new THREE.Vector3(.25,-.1,-1).normalize().multiplyScalar(radius*2.2));camera.lookAt(center);};
resetCamera();
const controls=[];
function datasetCamera(){
  camera.position.set(0,0,0);camera.up.set(0,-1,0);camera.lookAt(0,0,1);
  const [fx,fy,cx,cy]=reference.intrinsics,n=camera.near;
  camera.projectionMatrix.makePerspective(-cx/fx*n,(reference.width-cx)/fx*n,cy/fy*n,-(reference.height-cy)/fy*n,n,camera.far);
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();camera.updateMatrixWorld();
}
function makeHand(scene){
  const points=new THREE.InstancedMesh(sphereGeometry,new THREE.MeshBasicMaterial({depthTest:false}),21);
  points.frustumCulled=false;
  points.renderOrder=5;
  jointColors.forEach((c,i)=>points.setColorAt(i,new THREE.Color(c)));scene.add(points);
  const bones=colors.map(c=>{const b=new THREE.InstancedMesh(boneGeometry,new THREE.MeshBasicMaterial({color:c,depthTest:false}),4);b.frustumCulled=false;b.renderOrder=4;scene.add(b);return b;});
  function update(p){
    const visible=p && p.flat().every(Number.isFinite);points.visible=visible;bones.forEach(b=>b.visible=visible);if(!visible)return;
    p.forEach((x,i)=>points.setMatrixAt(i,new THREE.Matrix4().compose(new THREE.Vector3(...x),new THREE.Quaternion(),new THREE.Vector3(.0021,.0021,.0021))));
    points.instanceMatrix.needsUpdate=true;
    chains.forEach((chain,f)=>{
      for(let j=0;j<4;j++){
        let a=new THREE.Vector3(...p[chain[j]]),b=new THREE.Vector3(...p[chain[j+1]]),delta=b.clone().sub(a),length=delta.length();
        let q=new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0,1,0),delta.normalize());
        bones[f].setMatrixAt(j,new THREE.Matrix4().compose(a.add(b).multiplyScalar(.5),q,new THREE.Vector3(.0009,length,.0009)));
      }
      bones[f].instanceMatrix.needsUpdate=true;
    });
  }
  return {update};
}
function makeView(id,target){
  const element=$(id), scene=new THREE.Scene();scene.background=new THREE.Color(0xfafcff);
  scene.add(new THREE.HemisphereLight(0xffffff,0x77889c,2.7));
  const sun=new THREE.DirectionalLight(0xffffff,2.8);sun.position.set(1,-2,-2);scene.add(sun);
  const renderer=new THREE.WebGLRenderer({antialias:true,alpha:false});renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));element.appendChild(renderer.domElement);
  const control=new OrbitControls(camera,renderer.domElement);control.target.copy(center);control.enableDamping=false;control.minDistance=.08;control.maxDistance=4;controls.push(control);
  control.update();
  control.addEventListener('change',()=>{for(const other of controls)if(other!==control)other.target.copy(control.target);});
  const can=new THREE.Group();can.matrixAutoUpdate=false;
  if(canGeometry)can.add(new THREE.Mesh(canGeometry,new THREE.MeshStandardMaterial({color:0xb8c9db,metalness:.55,roughness:.4,side:THREE.DoubleSide})));
  const samples=new THREE.InstancedMesh(sphereGeometry,new THREE.MeshBasicMaterial({color:0x1e3a5f}),data.object_points.length);
  data.object_points.forEach((p,i)=>samples.setMatrixAt(i,new THREE.Matrix4().compose(new THREE.Vector3(...p),new THREE.Quaternion(),new THREE.Vector3(.0012,.0012,.0012))));can.add(samples);
  can.add(new THREE.AxesHelper(.04));scene.add(can);
  const hand=makeHand(scene);
  const meshes=target?data.geometry.map((shape,i)=>{
    const mesh=new THREE.Mesh(collisionGeometries[i],new THREE.MeshStandardMaterial({color:0x8399ae,metalness:.2,roughness:.65,transparent:true,opacity:.58,depthWrite:false}));
    mesh.matrixAutoUpdate=false;scene.add(mesh);return {mesh,link:shape.link};
  }):[];
  function render(){
    let w=element.clientWidth,h=element.clientHeight;
    if(renderer.domElement.width!==Math.round(w*renderer.getPixelRatio())||renderer.domElement.height!==Math.round(h*renderer.getPixelRatio()))renderer.setSize(w,h,false);
    if(reference && $('camera-mode').value==='dataset')datasetCamera();
    else{camera.aspect=w/h;camera.updateProjectionMatrix();}
    renderer.render(scene,camera);
  }
  return {scene,can,hand,meshes,render};
}
const views=[makeView('source',false),makeView('target',true)];
function selectCamera(){
  const calibrated=reference && $('camera-mode').value==='dataset';controls.forEach(c=>c.enabled=!calibrated);
  if(calibrated)datasetCamera();else{resetCamera();controls.forEach(c=>c.target.copy(center));}
}
selectCamera();
const duration=data.times.at(-1)||0;
let playing=false,position=0,lastClock=performance.now(),speed=1,renderCount=0,loopCount=0,current=null;
const visited=new Set();
$('frame').max=data.frame_ids.length-1;
$('heading').textContent=data.synthetic?'합성 FK 검증 동작':'실제 데이터 · 손–캔 리타게팅';
if(data.synthetic)$('source-title').textContent='입력 · 합성 FK reference와 캔';
$('summary').textContent=`원본 프레임 ${data.frame_ids[0]}–${data.frame_ids.at(-1)} · ${data.frame_ids.length}개 자세 · ${duration.toFixed(1)}초`+
  (data.time_basis==='retimed_user_authorized'?' · 느린 동작으로 시간 재배정 (촬영 fps와 별개)':data.timed?'':' · 확인용 재생 속도');
function drawAt(t){
  position=Math.max(0,Math.min(duration,t));
  let i=0;while(i<data.times.length-1 && data.times[i+1]<=position)i++;
  let j=Math.min(i+1,data.times.length-1),alpha=i===j?0:(position-data.times[i])/(data.times[j]-data.times[i]);
  // Never interpolate through a failed frame. Its stored candidate remains inspectable.
  if(!$('interpolate').checked || (alpha>0 && !data.transition_valid[j]))alpha=0;
  if(alpha===0)j=i;
  const human=alpha===0?data.human[i]:data.human[i].map((p,k)=>p.map((v,a)=>v+(data.human[j][k][a]-v)*alpha));
  const objectPose=objectPoses[i]&&objectPoses[j]?blendPose(objectPoses[i],objectPoses[j],alpha):null;
  views.forEach(v=>{v.can.visible=Boolean(objectPose);if(objectPose)v.can.matrix.copy(objectPose);});views[0].hand.update(human);
  let robot=null,links=null,full=null;
  const finite=wristPoses[i] && wristPoses[j] && data.q[i].every(x=>x!==null) && data.q[j].every(x=>x!==null);
  if(finite){
    const active=data.q[i].map((v,k)=>v+(data.q[j][k]-v)*alpha);
    ({points:robot,links,full}=fk(active,blendPose(wristPoses[i],wristPoses[j],alpha)));
  }
  views[1].hand.update(robot);
  for(const {mesh,link} of views[1].meshes){mesh.visible=Boolean(links)&&$('meshes').checked;if(links)mesh.matrix.copy(links[link]);}
  $('frame').value=i;$('frame-label').textContent=`${data.frame_ids[i]} / ${data.frame_ids.at(-1)}`;
  const ok=data.valid[i];$('status').className='status '+(ok?'good':'bad');
  const quality=data.quality?.[i];
  $('status').textContent=`${position.toFixed(2)} / ${duration.toFixed(2)}초 · `+(ok?'제약 검사 통과':'실패 또는 미검증')+
    (quality?` · 원본과 차이: 위치 ${quality.keypoint_rmse_mm.toFixed(1)} mm / 손가락 방향 ${quality.finger_direction_mean_deg.toFixed(1)}°`:'');
  current={index:i,frame:data.frame_ids[i],time:position,alpha,human,robot,full,objectMatrix:objectPose?.toArray()??null};visited.add(data.frame_ids[i]);
  drawReference(i);
}
function pause(){playing=false;$('play').textContent='▶ 재생';}
function play(){if(position>=duration)drawAt(0);playing=true;lastClock=performance.now();$('play').textContent='❚❚ 정지';}
$('play').onclick=()=>playing?pause():play();
$('restart').onclick=()=>{drawAt(0);lastClock=performance.now();};
$('frame').oninput=()=>{pause();drawAt(data.times[Number($('frame').value)]);};
$('speed').onchange=()=>{speed=Number($('speed').value);};
$('meshes').onchange=()=>drawAt(position);
$('interpolate').onchange=()=>drawAt(position);
$('rgb-points').onchange=()=>drawReference(current.index);
$('camera-mode').onchange=selectCamera;
$('reset-view').onclick=selectCamera;
drawAt(0);
function tick(now){
  if(playing){
    let next=position+(now-lastClock)/1000*speed;
    if(next>=duration){drawAt(duration);if($('loop').checked && duration>0){loopCount++;next%=duration;}else{next=duration;pause();}}
    drawAt(next);
  }
  lastClock=now;views.forEach(v=>v.render());renderCount++;requestAnimationFrame(tick);
}
requestAnimationFrame(tick);
// Small inspection API for deterministic playback/FK regression checks.
window.dexViewer={ready:!reference,play,pause,seekFrame:i=>{pause();drawAt(data.times[i]);},seekTime:t=>{pause();drawAt(t);},
  snapshot:()=>({...current,playing,renderCount,loopCount,visited:[...visited],referenceFrame:reference?data.frame_ids[current.index]:null,
    sameObject:views[0].can.matrix.equals(views[1].can.matrix)}),
  setLoop:value=>{$('loop').checked=Boolean(value);},setInterpolation:value=>{$('interpolate').checked=Boolean(value);drawAt(position);},
  projectSource:()=>{datasetCamera();return current.human.map(p=>{const q=new THREE.Vector3(...p).project(camera);return [(q.x+1)*reference.width/2,(1-q.y)*reference.height/2];});},
  resetHistory:()=>visited.clear(),frameCount:data.frame_ids.length,duration};
referenceLoaded.then(()=>{drawReference(current.index);window.dexViewer.ready=true;});
