import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';

const data=JSON.parse(document.getElementById('data').textContent);
const unit=new THREE.Vector3(1,1,1), $=id=>document.getElementById(id);
function pose(obj,p){obj.position.set(...p.slice(0,3));obj.quaternion.set(...p.slice(3,7));}
function surface(c){const g=new THREE.BufferGeometry();g.setAttribute('position',new THREE.Float32BufferAttribute(c.vertices.flat(),3));g.setIndex(c.faces.flat());g.computeVertexNormals();return g;}
const geometry=data.geometry.map(surface), sphere=new THREE.SphereGeometry(.002,8,6);
const canGeometries=data.collision_shapes.map(shape=>{const g=new THREE.CylinderGeometry(shape.radius,shape.radius,shape.height,128);g.rotateX(Math.PI/2);g.applyMatrix4(new THREE.Matrix4().set(...shape.transform.flat()));return g;});
const chains=['thumb','index','middle','ring','little'].map(f=>['wrist',...['mcp','pip','dip','tip'].map(j=>`${f}_${j}`)].map(n=>data.semantic_names.indexOf(n)));
const views=data.tracks.map((track,i)=>{
 const wrap=document.createElement('div');wrap.className='view';wrap.innerHTML=`<h3>${track.label}</h3><canvas></canvas><p class="status"></p>`;$('views').append(wrap);
 const canvas=wrap.querySelector('canvas'), renderer=new THREE.WebGLRenderer({canvas,antialias:true});renderer.setPixelRatio(Math.min(devicePixelRatio,2));
 const scene=new THREE.Scene();scene.background=new THREE.Color('#e8eef2');scene.add(new THREE.HemisphereLight(0xffffff,0x555555,2));
 const lamp=new THREE.DirectionalLight(0xffffff,2);lamp.position.set(1,-1,2);scene.add(lamp);
 const camera=new THREE.PerspectiveCamera(42,1,.005,10);camera.up.set(0,0,1);camera.position.set(.55,-.72,.57);
 const control=new OrbitControls(camera,canvas);control.target.set(0,0,.22);control.update();
 const grid=new THREE.GridHelper(1.2,24,0xa1aeb8,0xc7cfd6);grid.rotateX(Math.PI/2);scene.add(grid);
 const hand=data.geometry.map((c,k)=>{const mesh=new THREE.Mesh(geometry[k],new THREE.MeshStandardMaterial({color:i===0?0x507aab:0x60738b,metalness:.2,roughness:.6,transparent:true,opacity:i===0?.5:.95}));scene.add(mesh);return {mesh,index:track.body_names.indexOf(c.link)};});
 if(hand.some(h=>h.index<0))throw Error('Missing measured body for a collision mesh');
 const can=new THREE.Group(), canMaterial=new THREE.MeshStandardMaterial({color:0x089f94,metalness:.45,roughness:.4,transparent:true});canGeometries.forEach(g=>can.add(new THREE.Mesh(g,canMaterial)));scene.add(can);
 const humans=[];if(track.human){for(let k=0;k<21;k++){const m=new THREE.Mesh(sphere,new THREE.MeshBasicMaterial({color:0xe97424}));scene.add(m);humans.push(m);}}
 const bones=track.human?chains.map(()=>{const l=new THREE.Line(new THREE.BufferGeometry(),new THREE.LineBasicMaterial({color:0xe97424}));scene.add(l);return l;}):[];
 return {track,wrap,renderer,scene,camera,control,hand,can,canMaterial,humans,bones,index:0,ended:false};
});
let current=0,playing=false,last=performance.now(),renderCount=0;
$('slider').max=data.duration;$('slider').addEventListener('input',()=>{current=Number($('slider').value);playing=false;});
$('play').onclick=()=>{if(current>=data.duration)current=0;playing=!playing;};
function draw(){for(const v of views){
 const t=v.track;let j=0;while(j+1<t.times.length&&t.times[j+1]<=current)j++;v.index=j;v.ended=current>t.times.at(-1)+.015;
 v.hand.forEach(h=>{pose(h.mesh,t.links[j][h.index]);h.mesh.material.opacity=v.ended?.22:(t.human?.5:.95);});pose(v.can,t.object[j]);v.canMaterial.opacity=v.ended?.25:1;
 if(t.human){v.humans.forEach((m,k)=>m.position.set(...t.human[j][k]));v.bones.forEach((b,k)=>b.geometry.setFromPoints(chains[k].map(n=>new THREE.Vector3(...t.human[j][n]))));}
 v.wrap.querySelector('.status').textContent=t.human?t.status:`${v.ended?'기록 종료 · ':''}${t.status} · 마지막 ${t.times.at(-1).toFixed(2)}초`;
 const width=v.renderer.domElement.clientWidth,height=v.renderer.domElement.clientHeight;v.renderer.setSize(width,height,false);v.camera.aspect=width/height;v.camera.updateProjectionMatrix();v.control.update();v.renderer.render(v.scene,v.camera);
 }$('time').textContent=`${current.toFixed(2)} / ${data.duration.toFixed(2)}초`;$('slider').value=current;$('play').textContent=playing?'일시정지':'재생';renderCount++;}
function tick(now){if(playing){current=Math.min(data.duration,current+(now-last)/1000*Number($('speed').value));if(current>=data.duration)playing=false;}last=now;draw();requestAnimationFrame(tick);}requestAnimationFrame(tick);
window.residualViewer={seek:t=>{current=Math.max(0,Math.min(t,data.duration));draw();},play:()=>playing=true,pause:()=>playing=false,snapshot:()=>({time:current,renderCount,views:views.map(v=>({index:v.index,ended:v.ended,can:v.can.position.toArray(),lastTime:v.track.times.at(-1)}))}),ready:true};
