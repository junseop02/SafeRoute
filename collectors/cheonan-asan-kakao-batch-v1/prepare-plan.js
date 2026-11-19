'use strict';

const fs = require('fs');
const path = require('path');
const root = __dirname;
const points = JSON.parse(fs.readFileSync(path.join(root, 'config', 'locations.json'), 'utf8'));
const settings = JSON.parse(fs.readFileSync(path.join(root, 'settings.json'), 'utf8'));
const csvCell = (value) => {
  const text = String(value ?? '');
  return /[",\r\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
};
const row = (values) => values.map(csvCell).join(',') + '\n';
let csv = '\uFEFF' + row(['request_id','origin_city','origin_district','origin_name','destination_city','destination_district','destination_name','profile','priority','alternatives','avoid']);
let directedPairs = 0;
let requests = 0;
for (const origin of points) {
  for (const destination of points) {
    if (origin.id === destination.id) continue;
    directedPairs += 1;
    for (const profile of settings.directions_profiles) {
      requests += 1;
      csv += row([
        `${origin.id}--${destination.id}--${profile.id}`,
        origin.city, origin.district, origin.name,
        destination.city, destination.district, destination.name,
        profile.label, profile.priority, Boolean(profile.alternatives), profile.avoid || ''
      ]);
    }
  }
}
fs.writeFileSync(path.join(root, 'data', 'request_plan_preview.csv'), csv, 'utf8');
fs.writeFileSync(path.join(root, 'data', 'plan_manifest.json'), JSON.stringify({
  generated_at: new Date().toISOString(),
  regions: ['충청남도 천안시', '충청남도 아산시'],
  representative_points: points.length,
  directed_pairs: directedPairs,
  profiles: settings.directions_profiles.map((profile) => profile.id),
  planned_directions_requests: requests,
  note: '좌표와 실제 API 응답은 자동 수집기를 실행한 뒤 생성됩니다.'
}, null, 2));
console.log(`대표점 ${points.length}개, 방향성 ${directedPairs}쌍, 요청 계획 ${requests}건을 생성했습니다.`);
