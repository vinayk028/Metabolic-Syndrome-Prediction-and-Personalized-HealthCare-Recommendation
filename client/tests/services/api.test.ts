/**
 * API layer: auth header, 401 handling and request shapes.
 * Requests never leave the test — a fake axios adapter answers them.
 */
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from 'vitest';
import axios, { AxiosError, type AxiosAdapter, type InternalAxiosRequestConfig } from 'axios';
import type { PatientInfo } from '@/types/types';

type Api = typeof import('@/services/api');

const calls: InternalAxiosRequestConfig[] = [];
let status = 200;

const fakeAdapter: AxiosAdapter = async (config) => {
  calls.push(config);
  const response = { data: { ok: true }, status, statusText: '', headers: {}, config };
  if (status >= 400) throw new AxiosError('Request failed', String(status), config, null, response);
  return response;
};

const originalAdapter = axios.defaults.adapter;
let api: Api;

beforeAll(async () => {
  // api.ts creates its axios instance on import, copying these defaults
  axios.defaults.adapter = fakeAdapter;
  api = await import('@/services/api');
});
afterAll(() => {
  axios.defaults.adapter = originalAdapter;
});

beforeEach(() => {
  calls.length = 0;
  status = 200;
});

const lastCall = () => calls[calls.length - 1];

describe('auth header', () => {
  it('sends the saved token as a Bearer header', async () => {
    localStorage.setItem('token', 'abc123');
    await api.getAssessmentHistory();
    expect(lastCall().headers.Authorization).toBe('Bearer abc123');
  });

  it('sends no Authorization header for guests', async () => {
    await api.getMetabolicSyndromeNews();
    expect(lastCall().headers.Authorization).toBeUndefined();
  });
});

describe('401 handling', () => {
  it('signs the user out on 401 and passes the error on', async () => {
    window.history.replaceState(null, '', '/login'); // already on login: no redirect needed
    localStorage.setItem('token', 'expired');
    localStorage.setItem('user', '{"firstName":"A"}');
    status = 401;
    await expect(api.getAssessmentHistory()).rejects.toMatchObject({ response: { status: 401 } });
    expect(localStorage.getItem('token')).toBeNull();
    expect(localStorage.getItem('user')).toBeNull();
    expect(window.location.pathname).toBe('/login');
  });

  it('keeps the user signed in on other errors', async () => {
    localStorage.setItem('token', 'still-valid');
    status = 500;
    await expect(api.getAssessmentHistory()).rejects.toBeInstanceOf(AxiosError);
    expect(localStorage.getItem('token')).toBe('still-valid');
  });
});

describe('requests', () => {
  it('posts the prediction inputs as JSON', async () => {
    const info: PatientInfo = {
      age: 45,
      gender: 'Women',
      fattyLiver: false,
      hypertension: true,
      diabetes: false,
      systolicBP: 132,
      diastolicBP: 86,
      waistCircumference: 92,
    };
    await expect(api.predictMetabolicSyndrome(info)).resolves.toEqual({ ok: true });
    const c = lastCall();
    expect(c.method).toBe('post');
    expect(c.url).toBe('/mets/predict');
    expect(c.headers['Content-Type']).toBe('application/json');
    expect(JSON.parse(c.data)).toEqual(info);
  });

  it('sends only the fields the severity step needs', async () => {
    await api.calculateSeverity(
      { age: 45, gender: 'Men', fattyLiver: true, hypertension: false, diabetes: false, systolicBP: 140, diastolicBP: 90, waistCircumference: 104 },
      { fpg: 110, triglyceride: 180, hdlCholesterol: 38 },
      0.82,
    );
    expect(JSON.parse(lastCall().data)).toEqual({
      gender: 'Men',
      age: 45,
      systolicBP: 140,
      waistCircumference: 104,
      fpg: 110,
      triglyceride: 180,
      hdlCholesterol: 38,
      probability: 0.82,
    });
  });

  it('asks for news with limit and skip', async () => {
    await api.getMetabolicSyndromeNews(5, 15);
    expect(lastCall().url).toBe('/news?limit=5&skip=15');
    await api.getMetabolicSyndromeNews();
    expect(lastCall().url).toBe('/news?limit=15&skip=0');
  });

  it('uses the configured API base URL', async () => {
    await api.getMetabolicSyndromeNews();
    expect(lastCall().baseURL).toBe(import.meta.env.VITE_API_URL || 'http://localhost:5000/api');
  });
});

describe('downloadReport', () => {
  it('turns the base64 PDF into a file download', async () => {
    // jsdom has no object URLs; provide them for this test only
    const createObjectURL = vi.fn<(blob: Blob) => string>(() => 'blob:report');
    const revokeObjectURL = vi.fn();
    Object.assign(URL, { createObjectURL, revokeObjectURL });
    const click = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

    try {
      api.downloadReport(btoa('%PDF-1.4 test'), 'MetS-Report.pdf');

      const blob = createObjectURL.mock.calls[0][0];
      expect(blob.type).toBe('application/pdf');
      expect(await blob.text()).toBe('%PDF-1.4 test');
      const link = click.mock.contexts[0] as HTMLAnchorElement;
      expect(link.download).toBe('MetS-Report.pdf');
      expect(link.isConnected).toBe(false); // link removed afterwards
      expect(revokeObjectURL).toHaveBeenCalledWith('blob:report');
    } finally {
      delete (URL as Partial<typeof URL>).createObjectURL;
      delete (URL as Partial<typeof URL>).revokeObjectURL;
    }
  });
});
