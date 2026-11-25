const test=require('node:test');
const assert=require('node:assert/strict');
const fs=require('node:fs');const vm=require('node:vm');
const Novice=require('../legacy/core');
const candidate=(id,d,score=70,extra={})=>({route_id:id,distance_m:d,duration_s:600,road_count:10,guide_count:2,priority:'TIME',geometry:[[[127,36],[127.01,36]]],ai_pending:false,ai_eligible:true,easy_score:score,coverage:.9,...extra});
test('all caps, boundary, unlimited and highest easy score',()=>{
 const routes=[candidate('short',10000,60),candidate('1.5',15000,70),candidate('2',20000,80),candidate('3',30000,90),candidate('far',40000,99)];
 for(const [cap,id] of [[1.5,'1.5'],[2,'2'],[3,'3'],[Infinity,'far']]){
  const result=Novice.rankLive(routes,cap);assert.equal(result.recommended,id);assert.equal(result.routes.length,5);
 }
 assert.equal(routes[4].eligible,undefined);
});
test('partial failure, no candidates, pending and invalid scores',()=>{
 const routes=[candidate('low',10000,99,{coverage:.79}),candidate('failed',11000,null,{ai_eligible:false}),candidate('pending',12000,100,{ai_pending:true}),candidate('ok',14000,55)];
 assert.equal(Novice.rankLive(routes,1.5).recommended,'ok');
 assert.equal(Novice.rankLive(routes.slice(0,3),Infinity).recommended,null);
 assert.equal(Novice.rankLive([],2).recommended,null);
 assert.equal(Novice.rankLive([candidate('bad',0,99)],Infinity).recommended,null);
});
test('live UI reuses results, preserves selection, excludes detours, accepts late AI with current cap',async()=>{
 const elements=new Map();
 const element=()=>({textContent:'',children:[],classList:{toggle(){}},style:{},setAttribute(){},append(...x){this.children.push(...x)},replaceChildren(){this.children=[]},querySelectorAll(){return this.children},addEventListener(){}});
 const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
 $('cap').value='2';let calls=0;let finish;
 const context=vm.createContext({performance,Novice,$,document:{createElement:element},zoom:1,pan:[0,0],drawLiveRoute(){},setSearchStatus(){},console,fetch(){calls++;return new Promise(resolve=>finish=resolve)}});
 const app=fs.readFileSync(require('node:path').join(__dirname,'..','legacy','app.js'),'utf8');
 vm.runInContext(app.slice(app.indexOf('let liveRoute = null;'),app.indexOf("$('searchRoute').onclick = async")),context);
 context.input=[candidate('a',10000,60),candidate('b',18000,95)];
 vm.runInContext('liveRoutes=input;refreshLivePolicy();selectLiveRoute(1);',context);
 assert.equal(vm.runInContext('liveRecommendation',context),'b');
 $('cap').value='1.5';vm.runInContext('refreshLivePolicy()',context);
 assert.equal(vm.runInContext('liveRoute.route_id',context),'b');
 assert.equal(vm.runInContext('liveRecommendation',context),'a');
 assert.match($('reason').textContent,/우회 허용 범위 초과/);assert.match($('reason').textContent,/95.00/);assert.equal(calls,0);
 vm.runInContext('liveRoutes=input.map(r=>({...r,ai_pending:true,ai_eligible:false,easy_score:null}));refreshLivePolicy()',context);
 const promise=vm.runInContext("analyzeLiveRoutes([...liveRoutes],0,{name:'출발'},{name:'도착'})",context);
 $('cap').value='Infinity';vm.runInContext('refreshLivePolicy()',context);
 finish({ok:true,json:async()=>({success:true,results:[{route_id:'a',eligible:true,predicted_burden:40,easy_score:60,coverage:.9},{route_id:'b',eligible:true,predicted_burden:5,easy_score:95,coverage:.9}]})});
 await promise;assert.equal(calls,1);assert.equal(vm.runInContext('liveRecommendation',context),'b');assert.equal(vm.runInContext('liveRoutes.length',context),2);
 assert.match(app,/if \(liveRoute\) refreshLivePolicy\(\)/);
 assert.ok(!app.includes("$('cap').addEventListener('change'"));
});

test('detour selector is visible in live search before search button',()=>{
 const html=fs.readFileSync(require('node:path').join(__dirname,'..','legacy','index.html'),'utf8');
 const select=html.indexOf('<select id="cap"');
 assert.equal((html.match(/id="cap"/g)||[]).length,1);
 assert.ok(select>html.indexOf('id="destinationSearch"'));
 assert.ok(select<html.indexOf('id="searchRoute"'));
 assert.ok(select<html.indexOf('id="legacyRouteControls" hidden'));
 for(const value of ['1.5','2','3','Infinity']) assert.ok(html.includes('value="'+value+'"'));
});

function harness(){
 const elements=new Map(), pending=[];let status='';
 function element(){return {children:[],classList:{toggle(){}},style:{},setAttribute(){},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x},querySelectorAll(){return this.children},addEventListener(){}}}
 const $=id=>{if(!elements.has(id))elements.set(id,element());return elements.get(id)};
 $('cap').value='2';
 const context=vm.createContext({performance,Novice,$,document:{createElement:element},zoom:1,pan:[0,0],requestVersion:0,selected:null,draw(){},drawLiveRoute(){},setSearchStatus(v){status=v},console:{error(){},info(){}},URLSearchParams,selectedPlaces:{origin:{name:'A',longitude:127,latitude:36},destination:{name:'B',longitude:127.1,latitude:36.1}},fetch(url,options){return new Promise(resolve=>pending.push({url,options,resolve}))}});
 const app=fs.readFileSync(require('node:path').join(__dirname,'..','legacy','app.js'),'utf8');
 vm.runInContext(app.slice(app.indexOf('let liveRoute = null;'),app.indexOf('// 실제 경로 Canvas 표시')),context);
 return {context,$,pending,status:()=>status,run:s=>vm.runInContext(s,context)};
}
const response=data=>({ok:true,json:async()=>data});
const tick=()=>new Promise(resolve=>setImmediate(resolve));
const prediction=(id,score=70,extra={})=>({route_id:id,eligible:true,predicted_burden:100-score,easy_score:score,coverage:.9,...extra});
test('prediction IDs bind correctly even when result order changes or one fails',()=>{
 const routes=[candidate('a',1000),candidate('b',1100),candidate('c',1200)];
 const out=Novice.applyLivePredictions(routes,[prediction('c',91),{route_id:'b',eligible:false,error:'match failure'},prediction('a',61)]);
 assert.deepEqual(out.map(x=>x.easy_score),[61,null,91]);assert.equal(out[1].ai_error,'match failure');
 assert.equal(Novice.rankLive(out,2).recommended,'c');
});
test('missing duplicate malformed and inconsistent predictions cannot be recommended',()=>{
 for(const results of [[],[prediction('a',70),prediction('a',90)],[prediction('a',101)],[prediction('a',70,{easy_score:'70'})],[prediction('a',70,{predicted_burden:5})],[prediction('a',70,{coverage:.79})]]){
  const out=Novice.applyLivePredictions([candidate('a',1000)],results);
  assert.equal(out[0].ai_eligible,false);assert.equal(Novice.rankLive(out).recommended,null);assert.equal(out[0].easy_score,null);
 }
});
test('score direction, zero and hundred bounds and deterministic distance tie',()=>{
 assert.equal(Novice.rankLive([candidate('a',1000,0),candidate('b',1100,100)]).recommended,'b');
 assert.equal(Novice.rankLive([candidate('a',1000,70),candidate('b',1100,70)]).recommended,'a');
 assert.equal(Novice.rankLive([candidate('a',1000,101)]).recommended,null);
 assert.equal(Novice.rankLive([]).recommended,null);
});
test('late AI success or failure from previous search cannot overwrite newer routes',async()=>{
 for(const failure of [false,true]){
  const h=harness();h.context.input=[candidate('old',1000)];h.run('liveRoutes=input;refreshLivePolicy()');
  const old=h.run("analyzeLiveRoutes([...liveRoutes],0,{name:'old'},{name:'old'})");
  h.context.input=[candidate('new',2000,80)];h.run('liveRouteRequestVersion++;liveRoutes=input;refreshLivePolicy()');
  h.pending[0].resolve(failure?{ok:false,json:async()=>({error:'old failure'})}:response({success:true,results:[prediction('old',99)]}));
  await old;assert.equal(h.run('liveRoutes[0].route_id'),'new');assert.equal(h.run('liveRecommendation'),'new');
 }
});
test('partial and full AI failures are explicitly shown, no fallback recommendation',async()=>{
 const h=harness();h.context.input=[candidate('a',1000),candidate('b',1100)];h.run('liveRoutes=input;refreshLivePolicy()');
 let task=h.run("analyzeLiveRoutes([...liveRoutes],0,{name:'A'},{name:'B'})");
 h.pending[0].resolve(response({success:true,results:[prediction('a',75),{route_id:'b',eligible:false,error:'match failure'}]}));await task;
 assert.equal(h.run('liveRecommendation'),'a');assert.equal(h.run('liveRoutes[1].easy_score'),null);
 assert.match(h.status(),/1.*2/);
 task=h.run("analyzeLiveRoutes([...liveRoutes],0,{name:'A'},{name:'B'})");
 h.pending[1].resolve({ok:false,json:async()=>({error:'worker failed'})});await task;
 assert.equal(h.run('liveRecommendation'),null);assert.equal(h.run('liveRoutes.every(r=>r.easy_score===null)'),true);assert.match(h.status(),/실패/);
});
test('repeated searches show route first, ignore old analysis and bind current AI scores',async()=>{
 const h=harness();
 const a=h.$('searchRoute').onclick();h.pending[0].resolve(response({success:true,route:{},routes:[candidate('unused',1000)]}));await a;
 assert.equal(h.run('liveRoutes[0].ai_pending'),true);assert.equal(h.pending[1].url,'/api/predict-live');
 const b=h.$('searchRoute').onclick();h.pending[2].resolve(response({success:true,route:{},routes:[candidate('unused',2000)]}));await b;
 h.pending[1].resolve(response({success:true,results:[prediction('0',99)]}));await tick();
 assert.equal(h.run('liveRoutes[0].distance_m'),2000);assert.equal(h.run('liveRoutes[0].ai_pending'),true);
 h.pending[3].resolve(response({success:true,results:[prediction('0',65)]}));await tick();
 assert.equal(h.run('liveRoutes[0].easy_score'),65);assert.equal(h.run('liveRecommendation'),'0');
});
