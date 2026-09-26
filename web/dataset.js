import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const d=JSON.parse(document.getElementById('dataset').textContent), $=id=>document.getElementById(id);
const canvas=$('rgb'), ctx=canvas.getContext('2d');
[canvas.width,canvas.height]=d.image_size;
const images=d.images.map(path=>{const im=new Image();im.src=path;im.onload=()=>draw();return im;});
const boxes=d.frame_ids.map(()=>({left:null,right:null})), interpolated=d.frame_ids.map(()=>({left:false,right:false}));
let index=0,playing=false,last=null,phase=0,drag=null,cameraMode='source';
$('frame').max=d.frame_ids.length-1;
const scene=new THREE.Scene();scene.background=new THREE.Color('#fafcff');
const camera=new THREE.PerspectiveCamera(45,1,.001,100);camera.up.set(0,0,1);
camera.position.set(.6,-.8,.6);
const renderer=new THREE.WebGLRenderer({antialias:true});$('world').appendChild(renderer.domElement);
const controls=new OrbitControls(camera,renderer.domElement);
const grid=new THREE.GridHelper(1,20,0x617080,0xdde4ed);grid.rotation.x=Math.PI/2;grid.visible=false;scene.add(grid);
const axes=new THREE.AxesHelper(.1);axes.visible=false;scene.add(axes);
const group=new THREE.Group();scene.add(group);
scene.add(new THREE.HemisphereLight(0xffffff,0x68758c,2.2));
const light=new THREE.DirectionalLight(0xffffff,1.6);light.position.set(1,-2,3);scene.add(light);
const chains=Array.from({length:5},(_,f)=>[0,1+4*f,2+4*f,3+4*f,4+4*f]);
const good=p=>p&&p.every(x=>x!==null&&Number.isFinite(x));
$('show-meshes').disabled=!d.hand_meshes;
$('show-meshes').checked=Boolean(d.hand_meshes);
$('show-joints').disabled=!d.hand||Boolean(d.joints_are_proxies);
$('show-joints').checked=Boolean(d.hand)&&!d.joints_are_proxies;
$('show-projection').disabled=!d.object_overlay;
$('source-camera').disabled=!d.source_camera;
$('geometry').disabled=!d.object;
for(const option of $('geometry').options){
  if(option.value==='predicted')option.disabled=!d.prediction;
  if(option.value==='contact')option.disabled=!d.contact;
  if(option.value==='compare')option.disabled=!d.prediction&&!d.contact;
}
$('projection-note').textContent=d.source_camera?.assumed
  ?'카메라 K와 외부 좌표를 가정했습니다. RGB 정렬은 보정되지 않았습니다. 윤곽 표시는 입력 mesh의 투영 위치를 확인하는 용도입니다.'
  :d.source_camera?'윤곽은 입력 mesh의 볼록 외곽을 보정값으로 투영합니다. 영상에서 검출한 물체 경계가 아닙니다.':'RGB 입력만 표시합니다.';
if(d.joints_are_proxies)$('joints-note').textContent='보조점은 손 관절 측정값이 아니므로 표시하지 않습니다.';
const assumptions=Array.isArray(d.assumptions)?d.assumptions:[];
if(d.geometry_status==='test_assumptions'||assumptions.length){
  $('assumptions-panel').hidden=false;
  $('assumptions-title').textContent=d.geometry_status==='test_assumptions'?'테스트용 가정이 포함된 데이터':'입력에 사용한 가정';
  $('assumptions').textContent=assumptions.map(value=>`• ${typeof value==='string'?value:JSON.stringify(value)}`).join('\n')||'보정·좌표 정보에 테스트용 가정이 사용되었습니다.';
}
function inputCenter(){
  let values=d.object?.[index]?.filter(good)||[];
  if(!values.length&&d.hand_meshes)values=Object.values(d.hand_meshes).flatMap(mesh=>mesh.valid[index]?mesh.vertices[index].filter(good):[]);
  if(!values.length&&d.hand)values=d.hand[index].flat().filter(good);
  const center=new THREE.Vector3();for(const p of values)center.add(new THREE.Vector3(...p));
  return values.length?center.multiplyScalar(1/values.length):center;
}
function setCamera(mode){
  cameraMode=mode;const center=inputCenter(),source=d.source_camera;
  if(mode==='source'&&source?.valid?.[index]){
    const t=source.T_world_camera[index];camera.position.set(t[0][3],t[1][3],t[2][3]);
    camera.up.set(-t[0][1],-t[1][1],-t[2][1]);
    camera.fov=source.K?.[1]?.[1]>0?THREE.MathUtils.radToDeg(2*Math.atan(source.image_size[1]/(2*source.K[1][1]))):45;
    controls.target.copy(center);camera.lookAt(center);
    $('world-title').textContent=source.assumed?'원본 시선 방향 · RGB 보정 정렬 아님':'원본 카메라 위치에서 본 입력';
    $('view-note').textContent='입력 물체 중심을 바라보는 3D 검수 시점입니다. 화각은 K에서 계산하며, 영상과의 정확한 투영 일치를 뜻하지 않습니다.';
  }else{
    cameraMode='orbit';camera.up.set(0,0,1);camera.fov=45;
    camera.position.copy(center).add(new THREE.Vector3(.4,-.6,.4));controls.target.copy(center);camera.lookAt(center);
    $('world-title').textContent='전체 회전 시점 · metres';
    $('view-note').textContent='입력 좌표계를 자유롭게 회전합니다. 격자는 기본적으로 숨기며, 켜더라도 측정한 책상 표면을 뜻하지 않습니다.';
  }
  camera.updateProjectionMatrix();controls.update();
}
// User orbiting deliberately leaves the per-frame source-camera preset.
controls.addEventListener('start',()=>{cameraMode='orbit';});
function points(values,color,size,prob){
  const pos=[],colors=[];
  values.forEach((p,i)=>{if(!good(p))return;pos.push(...p);const c=new THREE.Color(color);
    if(prob&&prob[i]!==null)c.setHSL((1-Math.max(0,Math.min(1,prob[i])))*.65,1,.5);colors.push(c.r,c.g,c.b);});
  const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
  geometry.setAttribute('color',new THREE.Float32BufferAttribute(colors,3));
  group.add(new THREE.Points(geometry,new THREE.PointsMaterial({size,vertexColors:true})));
}
function world(){
  while(group.children.length){const child=group.children[0];group.remove(child);child.geometry.dispose();child.material.dispose();}
  const mode=$('geometry').value;
  if(d.object&&(mode==='input'||mode==='compare'))points(d.object[index],0x687887,.002);
  if(d.prediction?.available[index]&&(mode==='predicted'||mode==='compare'))points(d.prediction.points[index],0xec7c13,.003);
  if(d.hand_meshes&&$('show-meshes').checked){
    for(const [name,mesh] of Object.entries(d.hand_meshes)){
      if(!mesh.valid[index])continue;
      const vertices=mesh.vertices[index];if(!vertices?.length||!vertices.every(good))continue;
      const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(vertices.flat(),3));
      geom.setIndex(mesh.faces.flat());geom.computeVertexNormals();
      group.add(new THREE.Mesh(geom,new THREE.MeshStandardMaterial({color:name==='left'?0x805ad5:0x008878,roughness:.8,transparent:true,opacity:.5,side:THREE.DoubleSide,depthWrite:false})));
    }
  }
  if(d.hand&&$('show-joints').checked)d.hand[index].forEach((hand,side)=>{
    const color=side===0?0x805ad5:0x008878;points(hand,color,.006);
    for(const chain of chains){const pos=[];for(let j=1;j<chain.length;j++){
      const a=hand[chain[j-1]],b=hand[chain[j]];if(good(a)&&good(b))pos.push(...a,...b);}
      const geom=new THREE.BufferGeometry();geom.setAttribute('position',new THREE.Float32BufferAttribute(pos,3));
      group.add(new THREE.LineSegments(geom,new THREE.LineBasicMaterial({color})));
    }
  });
  if(d.contact?.available[index]&&(mode==='contact'||mode==='compare'))points(d.contact.points[index],0x2684e9,.003,mode==='contact'?d.contact.probability[index]:null);
  $('geometry-legend').textContent={input:'회색: 입력 물체. 예측 물체는 숨겨져 있습니다.',predicted:'주황: 네트워크 최종 pose로 변환한 물체. 손 표면은 입력 그대로입니다.',contact:'파랑→빨강: 접촉 단계 mesh의 예측 접촉 확률. 최종 pose와 다를 수 있습니다.',compare:'회색: 입력 물체 · 주황: 최종 예측 pose · 파랑: 접촉 단계 mesh.'}[mode];
  const warning=$('prediction-warning');warning.hidden=!d.contact||Boolean(d.contact.valid[index]);
  if(!warning.hidden)warning.textContent=!d.contact.available[index]?'이 프레임에는 네트워크 예측이 없습니다.':
    `이 프레임의 예측은 유효 라벨이 아닙니다. 정합성 검사: ${d.contact.consistency_pass?.[index]?'통과':'실패'}${d.source_camera?.assumed?' · 보정값은 테스트용 가정입니다.':''} 입력만 표시 중이어도 예측의 유효성이 바뀌지는 않습니다.`;
}
function draw(){
  const im=images[index];if(!im?.complete||!im.naturalWidth)return;
  ctx.clearRect(0,0,canvas.width,canvas.height);ctx.drawImage(im,0,0,canvas.width,canvas.height);
  if($('show-projection').checked&&d.object_overlay?.[index]?.length){
    const outline=d.object_overlay[index].filter(good);if(outline.length){
      ctx.strokeStyle='#ec7c13';ctx.lineWidth=3;ctx.setLineDash([8,5]);ctx.beginPath();ctx.moveTo(...outline[0]);
      for(const p of outline.slice(1))ctx.lineTo(...p);ctx.closePath();ctx.stroke();ctx.setLineDash([]);
    }
  }
  if($('show-joints').checked&&d.overlay?.[index])d.overlay[index].forEach((hand,side)=>{
    ctx.strokeStyle=side?'#00c9a0':'#ae78f1';ctx.lineWidth=2;
    for(const chain of chains){for(let j=1;j<chain.length;j++){
      const a=hand[chain[j-1]],b=hand[chain[j]];if(!good(a)||!good(b))continue;
      ctx.beginPath();ctx.moveTo(...a);ctx.lineTo(...b);ctx.stroke();}}
  });
  for(const side of ['left','right']){const box=boxes[index][side];if(!box)continue;
    ctx.strokeStyle=side==='right'?'#00dd88':'#b58aff';ctx.lineWidth=3;ctx.strokeRect(box[0],box[1],box[2]-box[0],box[3]-box[1]);}
  $('time').textContent=`Frame ${d.frame_ids[index]} · ${d.times[index].toFixed(3)} s`;
  let status=d.hand?`입력 존재 — 손 L/R: ${d.hand_valid[index].join(' / ')} · 물체: ${d.object_valid[index]} · 카메라: ${d.camera_valid[index]}\nGeometry provenance: ${d.geometry_status}`:'RGB 입력 영역을 지정할 수 있습니다. 3D 입력을 만든 후 build → review로 비교합니다.';
  if(d.contact)status+=`\n접촉 예측 있음: ${d.contact.available[index]} · 정합성 검사 통과: ${d.contact.consistency_pass?.[index]??d.contact.valid[index]} · 유효 라벨: ${d.contact.valid[index]}`;
  if(d.prediction?.available[index]){
    const pos=d.prediction.pose_difference_m?.[index],angle=d.prediction.pose_difference_deg?.[index],mesh=d.prediction.mesh_pose_error_m?.[index];
    if(Number.isFinite(pos))status+=`\n입력–최종 예측 위치 차이 ${(pos*1000).toFixed(1)} mm`;
    if(Number.isFinite(angle))status+=` · 회전 차이 ${angle.toFixed(1)}°`;
    if(Number.isFinite(mesh))status+=` · 접촉 mesh–최종 pose 최대 차이 ${(mesh*1000).toFixed(1)} mm`;
  }
  if(d.summary)status+=`\n손·물체 입력 프레임 ${d.summary.valid_frames}/${d.summary.frames} · 시간 기준: ${d.summary.time_basis}`;
  $('status').textContent=status;
}
function select(i){index=Math.max(0,Math.min(d.frame_ids.length-1,Number(i)));$('frame').value=index;draw();world();if(cameraMode==='source')setCamera('source');}
$('show-meshes').onchange=world;
$('show-joints').onchange=()=>{draw();world();};
$('show-projection').onchange=draw;
$('geometry').onchange=world;
$('show-grid').onchange=()=>{grid.visible=axes.visible=$('show-grid').checked;};
$('source-camera').onclick=()=>setCamera('source');
$('world-orbit').onclick=()=>setCamera('orbit');
$('frame').oninput=e=>{playing=false;select(e.target.value);};
$('play').onclick=()=>{playing=!playing;phase=d.times[index];last=null;$('play').textContent=playing?'정지':'재생';};
window.onkeydown=e=>{if(e.target.tagName==='INPUT'||e.target.tagName==='SELECT')return;if(e.key==='ArrowRight')select(index+1);if(e.key==='ArrowLeft')select(index-1);};
function pixel(e){const r=canvas.getBoundingClientRect();return [Math.max(0,Math.min(canvas.width,(e.clientX-r.left)*canvas.width/r.width)),Math.max(0,Math.min(canvas.height,(e.clientY-r.top)*canvas.height/r.height))];}
canvas.onpointerdown=e=>{if($('side').value==='none')return;playing=false;drag={point:pixel(e),side:$('side').value,index};canvas.setPointerCapture(e.pointerId);};
canvas.onpointerup=e=>{if(!drag)return;const a=drag.point,b=pixel(e);if(Math.abs(a[0]-b[0])>=2&&Math.abs(a[1]-b[1])>=2){boxes[drag.index][drag.side]=[Math.min(a[0],b[0]),Math.min(a[1],b[1]),Math.max(a[0],b[0]),Math.max(a[1],b[1])];interpolated[drag.index][drag.side]=false;}drag=null;draw();};
$('clear').onclick=()=>{if($('side').value==='none')return;boxes[index][$('side').value]=null;interpolated[index][$('side').value]=false;draw();};
$('previous').onclick=()=>{const s=$('side').value;if(s==='none'||index===0)return;boxes[index][s]=boxes[index-1][s]?.slice()??null;interpolated[index][s]=true;draw();};
$('interpolate').onclick=()=>{const s=$('side').value;if(s==='none')return;const anchors=boxes.map((b,i)=>b[s]&&!interpolated[i][s]?i:-1).filter(i=>i>=0);for(let j=1;j<anchors.length;j++){const a=anchors[j-1],b=anchors[j];for(let i=a+1;i<b;i++){const t=(d.times[i]-d.times[a])/(d.times[b]-d.times[a]);boxes[i][s]=boxes[a][s].map((x,k)=>x+(boxes[b][s][k]-x)*t);interpolated[i][s]=true;}}draw();};
$('save').onclick=()=>{const blob=new Blob([JSON.stringify({schema:1,capture_sha256:d.capture_sha256,image_size:d.image_size,frame_ids:d.frame_ids,boxes,interpolated},null,2)],{type:'application/json'});const url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='boxes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);};
$('load').onchange=async e=>{try{const value=JSON.parse(await e.target.files[0].text());if(value.capture_sha256!==d.capture_sha256||JSON.stringify(value.frame_ids)!==JSON.stringify(d.frame_ids)||JSON.stringify(value.image_size)!==JSON.stringify(d.image_size)||value.boxes.length!==boxes.length)throw Error('영상·프레임 정보가 다릅니다.');for(let i=0;i<boxes.length;i++){for(const s of ['left','right']){const b=value.boxes[i][s];if(b!==null&&(!Array.isArray(b)||b.length!==4||!good(b)))throw Error('잘못된 영역 값');} }for(let i=0;i<boxes.length;i++){boxes[i]=value.boxes[i];interpolated[i]=value.interpolated?.[i]??{left:false,right:false};}draw();}catch(error){$('status').textContent=error.message;}};
new ResizeObserver(()=>{const w=$('world').clientWidth,h=$('world').clientHeight;renderer.setSize(w,h);camera.aspect=w/h;camera.updateProjectionMatrix();}).observe($('world'));
function animate(now){requestAnimationFrame(animate);if(playing&&last!==null){phase+=(now-last)/1000;if(phase>d.times.at(-1))phase=d.times[0];let next=0;while(next+1<d.times.length&&d.times[next+1]<=phase)next++;if(next!==index)select(next);}last=now;controls.update();renderer.render(scene,camera);}
select(0);requestAnimationFrame(animate);
