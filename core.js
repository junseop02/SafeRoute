(function(root){
'use strict';
function score(r){return [r.accident_burden,r.narrow_burden,r.intersection_burden].every(x=>typeof x==='number'&&Number.isFinite(x))?100*(.55*r.accident_burden+.25*r.narrow_burden+.2*r.intersection_burden):null}
function rank(routes,cap=2){const shortest=Math.min(...routes.map(r=>r.distance_m).filter(x=>x>0));const all=routes.map(r=>({...r,score:score(r),eligible:r.coverage>=.8&&score(r)!==null&&r.distance_m<=shortest*cap}));all.sort((a,b)=>Number(b.eligible)-Number(a.eligible)||(a.score??Infinity)-(b.score??Infinity)||a.distance_m-b.distance_m);return {routes:all,recommended:all.filter(r=>r.eligible).length>=2?all.find(r=>r.eligible).route_id:null,shortest}}
function rankLive(routes, cap=2){
 const distances=routes.map(r=>r.distance_m).filter(d=>Number.isFinite(d)&&d>0);
 const shortest=distances.length?Math.min(...distances):null;
 const limit=cap===Infinity?Infinity:shortest*cap;
 const all=routes.map(r=>{
  const validDistance=Number.isFinite(r.distance_m)&&r.distance_m>0;
  const detourRatio=shortest&&validDistance?r.distance_m/shortest:null;
  const within=validDistance&&shortest!==null&&r.distance_m<=limit;
  const aiReady=!r.ai_pending&&r.ai_eligible===true&&Number.isFinite(r.easy_score)&&r.easy_score>=0&&r.easy_score<=100&&Number.isFinite(r.coverage)&&r.coverage>=.8;
  const reasons=[];
  if(!validDistance) reasons.push('거리 정보 없음');
  else if(!within) reasons.push('우회 허용 범위 초과');
  if(r.ai_pending) reasons.push('AI 분석 중');
  else if(!aiReady) reasons.push(r.ai_error||'AI 분석 실패 또는 도로 매칭률 80% 미만');
  return {...r,detour_ratio:detourRatio,eligible:within&&aiReady,exclusion_reason:reasons.join(' · ')};
 });
 const eligible=all.filter(r=>r.eligible).sort((a,b)=>b.easy_score-a.easy_score||a.distance_m-b.distance_m);
 return {routes:all,shortest,recommended:eligible[0]?.route_id??null};
}
function applyLivePredictions(routes, results){
 const byId=new Map(), duplicates=new Set();
 for(const result of results){
  if(!result || result.route_id === undefined || result.route_id === null) continue;
  const id=String(result.route_id);
  if(byId.has(id)) duplicates.add(id);
  byId.set(id,result);
 }
 return routes.map(route=>{
  const id=String(route.route_id), prediction=byId.get(id);
  let error=null;
  if(duplicates.has(id)) error='AI 응답 경로 ID 중복';
  else if(!prediction) error='이 경로의 AI 분석 결과가 없습니다.';
  else if(prediction.eligible!==true) error=prediction.error||'도로 매칭 기준 미충족';
  else if(!Number.isFinite(prediction.predicted_burden)||!Number.isFinite(prediction.easy_score)
    ||prediction.easy_score<0||prediction.easy_score>100||!Number.isFinite(prediction.coverage)) error='AI 분석 결과의 숫자가 올바르지 않습니다.';
  else if(prediction.coverage<.8) error='도로 매칭률 80% 미만';
  else if(Math.abs(prediction.easy_score-Math.max(0,Math.min(100,100-prediction.predicted_burden)))>.00011) error='AI 부담 점수와 쉬움 점수가 일치하지 않습니다.';
  return {...route,ai_pending:false,ai_eligible:!error,
   easy_score:error?null:prediction.easy_score,
   predicted_burden:error?null:prediction.predicted_burden,
   coverage:Number.isFinite(prediction?.coverage)?prediction.coverage:null,ai_error:error};
 });
}
const api={score,rank,rankLive,applyLivePredictions};if(typeof module!=='undefined')module.exports=api;else root.Novice=api;
})(typeof window==='undefined'?globalThis:window);
