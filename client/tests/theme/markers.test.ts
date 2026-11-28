import { describe, expect, it } from 'vitest';
import { buildMarkerRows } from '@/theme/markers';

const base = { gender: 'Men', systolicBP: 120, diastolicBP: 80, waistCircumference: 90 };

describe('buildMarkerRows', () => {
  it('only includes waist and blood pressure when no blood tests were entered', () => {
    const rows = buildMarkerRows(base);
    expect(rows.map((r) => r.key)).toEqual(['waist', 'bloodPressure']);
  });

  it('returns all five markers in a fixed order', () => {
    const rows = buildMarkerRows({ ...base, fpg: 95, triglyceride: 160, hdlCholesterol: 38 });
    expect(rows.map((r) => r.key)).toEqual(['waist', 'bloodPressure', 'glucose', 'triglycerides', 'hdl']);
  });

  it('shows blood pressure as systolic/diastolic and maps status to risk', () => {
    const rows = buildMarkerRows({ ...base, fpg: 95, triglyceride: 160, hdlCholesterol: 38 });
    const bp = rows.find((r) => r.key === 'bloodPressure')!;
    expect(bp.display).toBe('120/80');
    expect(bp.unit).toBe('mmHg');
    expect(bp.status).toBe('borderline');
    expect(bp.risk).toBe('medium');

    const byKey = Object.fromEntries(rows.map((r) => [r.key, r.risk]));
    expect(byKey).toMatchObject({ waist: 'low', glucose: 'medium', triglycerides: 'high', hdl: 'high' });
  });

  it('builds ascending range zones that end at the bar maximum', () => {
    for (const row of buildMarkerRows({ ...base, fpg: 95, triglyceride: 160, hdlCholesterol: 38 })) {
      const tos = row.zones.map((z) => z.to);
      expect([...tos].sort((a, b) => a - b)).toEqual(tos);
      expect(tos[tos.length - 1]).toBe(row.max);
      expect(row.min).toBeLessThan(row.max);
    }
  });

  it('moves the waist and HDL zones for women', () => {
    const men = buildMarkerRows({ ...base, hdlCholesterol: 45 });
    const women = buildMarkerRows({ ...base, gender: 'Women', hdlCholesterol: 45 });
    expect(men.find((r) => r.key === 'waist')!.zones[1].to).toBe(102);
    expect(women.find((r) => r.key === 'waist')!.zones[1].to).toBe(88);
    expect(men.find((r) => r.key === 'hdl')!.status).toBe('borderline');
    expect(women.find((r) => r.key === 'hdl')!.status).toBe('elevated');
  });

  it('skips markers whose value is not a number (e.g. a cleared field)', () => {
    const rows = buildMarkerRows({ ...base, fpg: NaN, diastolicBP: NaN });
    expect(rows.some((r) => r.key === 'glucose')).toBe(false);
    expect(rows.some((r) => r.key === 'bloodPressure')).toBe(false);
  });
});
