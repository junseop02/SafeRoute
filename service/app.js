'use strict';
const $=id=>document.getElementById(id);
const state={origin:null,destination:null,result:null,selected:null,map:null,markers:{},lines:[],requestId:0,envKey:false};
const pretty={RECOMMEND:'카카오 추천',TIME:'카카오 시간 우선',DISTANCE:'카카오 거리 우선',SCHOOLZONE:'어린이보호구역 회피 후보',UTURN:'유턴 회피 후보',TOLL:'유료도로 회피 후보',MOTORWAY:'고속도로 회피 후보'};
function status(message,error=false){$('status').textContent=message;$('status').classList.toggle('error',error)}
async function api(endpoint,body){const response=await fetch(endpoint,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({...body,key:$('apiKey').value.trim()})});let data;try{data=await response.json()}catch{throw new Error('로컬 서버 응답을 읽지 못했습니다.')}if(!response.ok)throw new Error(data.error||'요청 실패');return data}
function latLng(place){return new kakao.maps.LatLng(Number(place.y),Number(place.x))}
function clearLines(){for(const line of state.lines)line.setMap(null);state.lines=[]}
function showMarker(kind,place){if(!state.map)return;if(state.markers[kind])state.markers[kind].setMap(null);state.markers[kind]=new kakao.maps.Marker({map:state.map,position:latLng(place),title:(kind==='origin'?'출발: ':'도착: ')+place.name})}
function fitMap(onlySelected=false){if(!state.map)return;state.map.relayout();const bounds=new kakao.maps.LatLngBounds();let count=0;
  const routes=state.result?.routes||[];for(const route of (onlySelected&&state.selected?[state.selected]:routes)){for(const line of route.geometry){for(const p of line){if(Number.isFinite(p[0])&&Number.isFinite(p[1])){bounds.extend(new kakao.maps.LatLng(p[1],p[0]));count++}}}}
  if(!count){for(const place of [state.origin,state.destination])if(place){bounds.extend(latLng(place));count++}}
  if(count>1)state.map.setBounds(bounds,50,50,50,50);else if(count===1){state.map.setCenter(latLng(state.origin||state.destination));state.map.setLevel(5)}
}
function drawMap(){if(!state.map)return;clearLines();for(const kind of ['origin','destination']){if(state[kind])showMarker(kind,state[kind]);else if(state.markers[kind]){state.markers[kind].setMap(null);delete state.markers[kind]}}
  if(state.result){const ordered=[...state.result.routes.filter(r=>r.id!==state.selected?.id),...(state.selected?[state.selected]:[])];for(const route of ordered){const active=route.id===state.selected?.id;for(const line of route.geometry){if(line.length<2)continue;const poly=new kakao.maps.Polyline({map:state.map,path:line.map(p=>new kakao.maps.LatLng(p[1],p[0])),strokeWeight:active?6:3,strokeColor:active?'#087ba8':'#879b9e',strokeOpacity:active?.9:.65});state.lines.push(poly)}}}
  fitMap(Boolean(state.result&&state.selected));
}
function loadMap(){const key=$('mapKey').value.trim();if(!key){$('mapNotice').textContent='카카오 JavaScript 키를 입력하세요.';return}if(state.map){drawMap();return}
  const button=$('loadMap');button.disabled=true;$('mapNotice').hidden=false;$('mapNotice').textContent='카카오 지도를 불러오는 중입니다…';
  const ready=()=>{try{kakao.maps.load(()=>{state.map=new kakao.maps.Map($('map'),{center:new kakao.maps.LatLng(36.8,127.5),level:12});$('mapNotice').hidden=true;button.disabled=false;drawMap()})}catch(error){$('mapNotice').textContent='지도를 열지 못했습니다. JavaScript 키와 등록된 SDK 도메인을 확인하세요.';button.disabled=false}};
  if(window.kakao?.maps?.load){ready();return}
  const script=document.createElement('script');script.src='https://dapi.kakao.com/v2/maps/sdk.js?appkey='+encodeURIComponent(key)+'&autoload=false';script.onload=ready;script.onerror=()=>{$('mapNotice').textContent='지도 SDK를 불러오지 못했습니다. 인터넷 연결·JavaScript 키·등록 도메인을 확인하세요.';button.disabled=false;script.remove()};document.head.append(script);
}
function resetRoutes(){state.requestId++;$('findRoutes').disabled=false;state.result=null;state.selected=null;$('recommendations').hidden=true;$('cards').replaceChildren();$('selectedScore').replaceChildren();$('metrics').replaceChildren();$('detail').textContent='';$('export').hidden=true;$('routeTitle').textContent='카카오 지도';drawMap()}
function selectedPlace(kind,place){state[kind]=place;$(kind+'Picked').textContent=`선택: ${place.name} · ${place.address}`;$(kind+'Picked').classList.add('ok');$(kind+'Query').value=place.name;$(kind+'Results').replaceChildren();resetRoutes();if(state.map)fitMap();status('출발지와 도착지가 선택되면 경로 후보를 찾으세요.')}
async function search(kind){const query=$(kind+'Query').value.trim();if(query.length<2){status('장소 이름이나 주소를 2글자 이상 입력하세요.',true);return}const button=$(kind+'Search');button.disabled=true;status('카카오에서 장소를 검색 중입니다…');try{const data=await api('/api/places',{query});const box=$(kind+'Results');box.replaceChildren();if(!data.places.length){status('검색 결과가 없습니다. 장소 이름이나 주소를 더 자세히 입력하세요.',true);return}for(const place of data.places){const b=document.createElement('button');b.type='button';b.className='result';const name=document.createElement('strong'),address=document.createElement('small');name.textContent=place.name;address.textContent=place.address;b.append(name,address);b.onclick=()=>selectedPlace(kind,place);box.append(b)}status(`검색 결과 ${data.places.length}개. 알맞은 장소를 선택하세요.`)}catch(error){status(error.message,true)}finally{button.disabled=false}}
function queryChanged(kind){if(state[kind]){state[kind]=null;$(kind+'Picked').textContent='장소를 선택하세요';$(kind+'Picked').classList.remove('ok');resetRoutes()}}
async function findRoutes(){if(!state.origin||!state.destination){status('출발지와 도착지의 검색 결과를 각각 선택하세요.',true);return}resetRoutes();const requestId=++state.requestId;const button=$('findRoutes');button.disabled=true;$('recommendations').hidden=false;$('scoreIntro').textContent='카카오 경로 후보를 찾고 점수를 계산 중입니다…';$('cards').replaceChildren();status('선택한 우회 범위로 카카오 경로를 다시 찾고 모델 점수를 계산 중입니다…');try{const result=await api('/api/routes',{origin:state.origin,destination:state.destination,detour_cap:$('detourCap').value});if(requestId!==state.requestId)return;state.result=result;state.selected=result.routes.find(r=>r.id===result.recommended)||result.routes[0]||null;render();$('recommendations').scrollIntoView?.({behavior:'smooth',block:'start'});if(result.recommended)status(`우회 범위 안의 서로 다른 경로 ${result.routes.length}개 중 모델 비교 가능 ${result.distinct_eligible}개. P01-v1.1 정책에 따른 추천 경로를 표시합니다.`);else if(result.routes.length<2)status(`우회 범위 안의 서로 다른 경로 ${result.routes.length}개만 찾았습니다. AI 추천에는 비교 가능한 경로가 최소 2개 필요합니다. 우회 범위를 넓혀 보세요.`,true);else status(`경로 ${result.routes.length}개를 표시합니다. 모델 비교 가능 경로가 ${result.distinct_eligible}개입니다. 최소 2개가 필요하므로 추천을 보류합니다.`,true);$('export').hidden=false}catch(error){if(requestId!==state.requestId)return;resetRoutes();status(error.message,true)}finally{if(requestId===state.requestId)button.disabled=false}}
function percent(v){return v==null?'자료 없음':`${(v*100).toFixed(1)}%`}
function easyScore(route) {

    const burden = route.model_score;

    if (
        !Number.isFinite(burden) ||
        burden < 0 ||
        burden > 100 ||
        !Number.isFinite(route.model_coverage) ||
        route.model_coverage < 0.8 ||
        route.model_coverage > 1 ||
        route.model_version !== 'T09' ||
        route.policy_version !== 'P01'
    ) {
        return null;
    }

    return 100 - burden;
}
function scoreLabel(route){const score=easyScore(route);return score==null?'계산 불가':`${score.toFixed(1)}`}
function element(tag,className,text){const item=document.createElement(tag);if(className)item.className=className;if(text!=null)item.textContent=text;return item}
function render(){const data=state.result;if(!data)return;
  const scored=data.routes.filter(r=>easyScore(r)!=null);
  $('scoreIntro').textContent=scored.length?`경로 추천 점수는 높을수록 P01 정책에서 우선하는 경로입니다. ${data.routes.length}개 후보 중 ${scored.length}개를 점수화했습니다.`:'점수를 계산할 수 있는 경로가 없습니다. 도로 자료 매칭률을 확인하세요.';
  $('cards').replaceChildren();
  for(const route of data.routes){
    const card=element('button','card scoreCard'+(state.selected?.id===route.id?' selected':''));
    card.type='button';card.setAttribute('aria-pressed',String(state.selected?.id===route.id));
    const head=element('div','scoreCardHead');
    head.append(element('strong','scoreCardTitle',pretty[route.profile]||route.profile));
    if(data.recommended===route.id)head.append(element('span','scoreBadge','AI 추천'));
    else if(route.eligible)head.append(element('span','compareBadge','비교 가능'));
    card.append(head);
    const scoreRow=element('div','scoreRow');
    scoreRow.append(element('span','scoreCaption','경로 추천 점수'));
    const score=element('span','scoreNumber',scoreLabel(route));
    if(easyScore(route)!=null)score.append(element('small','scoreUnit',' / 100점'));
    scoreRow.append(score);card.append(scoreRow);
    const bar=element('div','scoreBar');bar.setAttribute('role','progressbar');bar.setAttribute('aria-label','경로 추천 점수');
    bar.setAttribute('aria-valuemin','0');bar.setAttribute('aria-valuemax','100');
    if(easyScore(route)!=null)bar.setAttribute('aria-valuenow',scoreLabel(route));
    const fill=element('span','scoreBarFill');fill.style.width=(easyScore(route)??0)+'%';bar.append(fill);card.append(bar);
    const stats=element('div','scoreStats');
    const detour=data.shortest_m>0?route.distance_m/data.shortest_m:null;
    for(const [value,label] of [[(route.distance_m/1000).toFixed(2)+' km','이동 거리'],[Math.round(route.duration_s/60)+'분','예상 시간'],[detour==null?'자료 없음':detour.toFixed(2)+'배','우회 비율']]){
      const stat=element('span','scoreStat');stat.append(element('b','',value),element('small','',label));stats.append(stat)
    }
    card.append(stats);
    card.append(element('small','scoreNote',easyScore(route)==null?`도로 매칭률 ${percent(route.model_coverage)} · 점수 계산 불가`:`도로 매칭률 ${percent(route.model_coverage)} · ${data.recommended===route.id?'현재 후보 중 추천 조건 충족':'점수가 높을수록 P01 정책에서 우선'}`));
    card.onclick=()=>{state.selected=route;render()};$('cards').append(card)
  }
  $('routeTitle').textContent=`${data.origin.name} → ${data.destination.name}`;
  const route=state.selected;if(!route){$('selectedScore').replaceChildren();$('metrics').replaceChildren();$('detail').textContent='표시할 경로가 없습니다.';drawMap();return}
  const selected=element('div','selectedScoreBox');
  selected.append(element('span','','선택 경로의 경로 추천 점수'));
  selected.append(element('b','',easyScore(route)==null?'계산 불가':`${scoreLabel(route)} / 100점`));
  selected.append(element('small','',easyScore(route)==null?'도로 자료가 충분히 매칭되지 않았습니다.':'높을수록 P01 정책에서 우선하는 경로 · 실제 사고 확률이 아닙니다.'));
  $('selectedScore').replaceChildren(selected);
  const metrics=[[percent(route.model_wide_ratio),'차로 2개 이상 길이 비중'],[route.model_intersections==null?'자료 없음':route.model_intersections.toFixed(1),'주변 200m 교차로 수 가중 평균'],[percent(route.model_hotspot_ratio),'사고다발지 150m 이내 길이 비중(참고)'],[percent(route.model_coverage),'모델 매칭률']];
  $('metrics').replaceChildren(...metrics.map(([value,label])=>{const div=element('div','metric');div.append(element('b','',value),element('small','',label));return div}));
  $('detail').textContent=easyScore(route)==null?'도로 매칭률 80% 이상 및 유효한 T09/P01 점수 조건을 충족하지 않아 점수를 표시하지 않습니다.':`경로 추천 점수 = 100 − 모델 부담 점수. T09 모델 점수 80%, 좁은 도로 부담 10%, 교차로 부담 10%를 결합한 상대 비교용 지표입니다. 예상 시간 약 ${Math.round(route.duration_s/60)}분은 점수 계산에 사용하지 않으며 부담 점수와 이동 거리가 같을 때 비교합니다.`;
  drawMap();
}
for(const kind of ['origin','destination']){$(kind+'Search').onclick=()=>search(kind);$(kind+'Query').onkeydown=e=>{if(e.key==='Enter')search(kind)};$(kind+'Query').oninput=()=>queryChanged(kind)}
$('loadMap').onclick=loadMap;$('findRoutes').onclick=findRoutes;$('detourCap').onchange=()=>{if(state.origin&&state.destination)findRoutes()};
$('zoomIn').onclick=()=>state.map?.setLevel(Math.max(1,state.map.getLevel()-1));$('zoomOut').onclick=()=>state.map?.setLevel(Math.min(14,state.map.getLevel()+1));$('resetView').onclick=()=>fitMap();
$('export').onclick=()=>{if(!state.result)return;const blob=new Blob([JSON.stringify(state.result,null,2)],{type:'application/json'}),url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download='saferoute_kakao_model_routes.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
if(location.protocol==='file:')status('전국 검색은 로컬 서버가 필요합니다. START_SAFEROUTE.cmd를 실행하세요.',true);else fetch('/api/status').then(r=>r.json()).then(s=>{state.envKey=s.api_key_from_environment;if(state.envKey){$('apiKey').placeholder='실행 환경에 키가 설정되어 있습니다';$('keyHint').textContent='서버 환경 변수를 사용합니다. 이 화면에 키를 입력할 필요가 없습니다.'}}).catch(()=>status('로컬 서버 상태를 확인하지 못했습니다.',true));
