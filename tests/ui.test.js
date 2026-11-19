const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),vm=require('node:vm'),path=require('node:path');
const app=fs.readFileSync(path.join(__dirname,'..','app.js'),'utf8');
function setup(){
 function element(tag){return {tag,children:[],attrs:{},style:{},textContent:'',setAttribute(k,v){this.attrs[k]=v},append(...x){this.children.push(...x)},replaceChildren(...x){this.children=x},addEventListener(k,fn){this[k]=fn}}}
 const nodes=new Map();const $=id=>{if(!nodes.has(id))nodes.set(id,element('div'));return nodes.get(id)};
 const context=vm.createContext({$,document:{createElement:element},liveRoutes:[],liveRecommendation:null,selectLiveRoute(i){context.selection=i}});
 vm.runInContext(app.slice(app.indexOf('function routeCardView('),app.indexOf('// 카카오 경로 조회 완료 후 AI 분석')),context);
 return {context,$,run:s=>vm.runInContext(s,context)};
}
const route=(id,extra={})=>({route_id:id,priority:'TIME',distance_m:10000,duration_s:600,detour_ratio:1.2,road_count:10,guide_count:11,eligible:true,easy_score:85.4,coverage:.9,...extra});
test('pending and failed cards never display a fake zero score',()=>{const h=setup();for(const item of [route('a',{ai_pending:true,easy_score:null}),route('a',{eligible:false,easy_score:null,ai_error:'timeout'})]){h.context.input=item;const view=h.run('routeCardView(input,null)');assert.equal(view.hasScore,false);assert.ok(['分析','분석'].some(x=>view.scoreText.includes(x)));assert.notEqual(view.scoreText,'0.0');}});
test('high score beyond detour stays excluded and explanation is retained',()=>{const h=setup();h.context.input=route('a',{eligible:false,easy_score:99,exclusion_reason:'우회 허용 범위 초과'});const before=JSON.stringify(h.context.input);const view=h.run('routeCardView(input,"a")');assert.equal(view.badge,'추천 제외');assert.match(view.explanation,/우회/);assert.equal(view.scoreText,'99.0');assert.equal(JSON.stringify(h.context.input),before);});
test('cards are keyboard buttons, labeled, and retain click to select routes',()=>{const h=setup();h.context.liveRoutes=[route('a'),route('b',{eligible:false,easy_score:null})];h.context.liveRecommendation='a';h.run('renderLiveRouteCards()');const cards=h.$('cards').children;assert.equal(cards.length,2);assert.equal(cards[0].tag,'button');assert.equal(cards[0].type,'button');assert.match(cards[0].attrs['aria-label'],/AI 추천/);assert.match(cards[1].attrs['aria-label'],/분석 불가/);cards[1].click();assert.equal(h.context.selection,1);});
test('no eligible candidates are clearly explained in summary',()=>{const h=setup();h.context.liveRoutes=[route('a',{eligible:false,easy_score:null})];h.run('renderLiveRouteCards()');assert.match(h.$('routeSummary').textContent,/추천 가능한 경로가 없습니다/);});
