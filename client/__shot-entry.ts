/**
 * Temporary audit entry (deleted after stage 6): boots the real app with a fake
 * API so every page renders with data. Query: ?route=/dashboard&auth=1&theme=dark&click=sel1|sel2
 */
import axios, { type AxiosAdapter } from 'axios';

const q = new URLSearchParams(location.search);
const route = q.get('route') || '/';
const theme = q.get('theme') || 'light';
localStorage.setItem('mui-mode', theme);
document.documentElement.setAttribute('data-theme', theme);

const user = {
  id: 'u1', firstName: 'Asha', lastName: 'Rao', fullName: 'Asha Rao', email: 'asha@example.org',
  phone: '9876543210', gender: 'Women', dateOfBirth: '1984-05-02', role: 'user', createdAt: '2025-01-10T00:00:00Z',
};
if (q.get('auth') === '1') {
  localStorage.setItem('token', 'audit-token');
  localStorage.setItem('user', JSON.stringify(user));
} else {
  localStorage.removeItem('token');
  localStorage.removeItem('user');
}

const recs = {
  dietPlan: ['Vegetables at every meal', 'Whole grains instead of refined', 'Oily fish twice a week'],
  avoidList: ['Sugary drinks', 'Deep-fried snacks'],
  exercisePlan: ['30 min brisk walk, 5 days a week', 'Strength training twice a week'],
  yogaPoses: ['Tadasana', 'Bhujangasana'],
};
const params = (sys: number, dia: number, waist: number, fpg: number, tg: number, hdl: number) => ({
  age: 42, gender: 'Women', fattyLiver: false, hypertension: false, diabetes: false,
  systolicBP: sys, diastolicBP: dia, waistCircumference: waist, fpg, triglyceride: tg, hdlCholesterol: hdl,
});
const pastChecks = [
  { _id: 'a1', date: '2026-03-02T09:00:00Z', probability: 0.81, severity: 2.4, riskLevel: 'High Severity', inputParameters: params(142, 92, 98, 118, 190, 42), recommendations: recs },
  { _id: 'a2', date: '2026-06-04T09:00:00Z', probability: 0.64, severity: 1.2, riskLevel: 'Medium Severity', inputParameters: params(134, 86, 94, 106, 165, 46), recommendations: recs },
  { _id: 'a3', date: '2026-09-08T09:00:00Z', probability: 0.38, severity: 0.4, riskLevel: 'Low Severity', inputParameters: params(126, 82, 90, 98, 140, 52), recommendations: recs },
];
const articles = Array.from({ length: 6 }, (_, i) => ({
  title: `Study ${i + 1}: daily walking lowers metabolic risk markers`,
  description: 'Researchers followed adults for two years and found steady improvements in waist size and blood sugar.',
  content: '', url: `https://example.org/news/${i + 1}`, image: null, source: 'Example Health News',
  publishedAt: `2026-10-0${i + 1}T00:00:00Z`, author: null,
}));

const reply = (url: string, method: string): unknown => {
  if (url.startsWith('/auth/me')) return { success: true, user };
  if (url.startsWith('/auth/assessments')) return { success: true, assessmentHistory: pastChecks };
  if (url.startsWith('/news')) return { success: true, articles, count: articles.length };
  if (url.startsWith('/mets/predict')) return { probability: 0.78, hasMetabolicSyndrome: true };
  if (url.startsWith('/mets/severity')) return { severity: 2.1, riskLevel: 'High Severity' };
  if (url.startsWith('/mets/recommendations')) return recs;
  if (method !== 'get') return { success: true, user, message: 'ok' };
  return { success: true };
};
const fake: AxiosAdapter = async (config) => ({
  data: reply(config.url || '', (config.method || 'get').toLowerCase()), status: 200, statusText: 'OK', headers: {}, config,
});
axios.defaults.adapter = fake;

history.replaceState(null, '', route);

// Finish animations instantly (headless Chrome freezes them), then optional clicks
const freeze = document.createElement('style');
freeze.textContent = '*,*::before,*::after{transition:none!important;animation:none!important}[style*="opacity"]{opacity:1!important}';
document.head.appendChild(freeze);
(q.get('click') || '').split('|').filter(Boolean).forEach((sel, i) =>
  setTimeout(() => (document.querySelector(sel) as HTMLElement | null)?.click(), 1500 + i * 900),
);

await import('./src/main.tsx');
