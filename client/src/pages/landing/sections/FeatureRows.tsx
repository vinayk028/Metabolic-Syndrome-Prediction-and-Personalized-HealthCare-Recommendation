/**
 * FeatureRows — three alternating text + UI-snippet rows, 
 */
import type { ReactNode } from 'react';
import { Container, Typography } from '@mui/material';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import RestaurantRoundedIcon from '@mui/icons-material/RestaurantRounded';
import DirectionsRunRoundedIcon from '@mui/icons-material/DirectionsRunRounded';
import SelfImprovementRoundedIcon from '@mui/icons-material/SelfImprovementRounded';
import BlockRoundedIcon from '@mui/icons-material/BlockRounded';
import { AnimatedReveal, RiskBadge } from '../../../components/ui';

/* ── Snippets ─────────────────────────────────────────────────────────── */

const PipelineSnippet = () => (
  <div className="lp-snippet lp-pipe" role="img" aria-label="How it works: your markers go in, you get a 52% probability and a moderate severity, then a personal plan.">
    {[
      { k: 'Input', v: '5 markers + age, sex, history' },
      { k: 'Prediction', v: 'Probability 52%' },
      { k: 'Severity Calculation', v: 'Severity: Moderate', badge: true },
      { k: 'Output', v: 'Personal plan & PDF report' },
    ].map((s, i, arr) => (
      <div key={s.k} className="lp-pipe__step" aria-hidden>
        <span className="lp-pipe__dot">{i + 1}</span>
        <div className="lp-pipe__card">
          <span className="lp-pipe__k">{s.k}</span>
          <span className="lp-pipe__v">
            {s.v}
            {s.badge && <RiskBadge level="medium" label="Moderate" size="small" />}
          </span>
        </div>
        {i < arr.length - 1 && <span className="lp-pipe__line" />}
      </div>
    ))}
  </div>
);

const PlanSnippet = () => (
  <div className="lp-snippet lp-plan" role="img" aria-label="Sample personalised plan with diet, exercise and yoga tabs.">
    <div className="lp-plan__tabs" aria-hidden>
      <span className="is-active">
        <RestaurantRoundedIcon /> Diet
      </span>
      <span>
        <DirectionsRunRoundedIcon /> Exercise
      </span>
      <span>
        <SelfImprovementRoundedIcon /> Yoga
      </span>
    </div>
    <ul className="lp-plan__list" aria-hidden>
      {['Oats with berries and walnuts for breakfast', 'Half your plate as non-starchy vegetables', 'Lentils or beans 3–4 times a week', 'Olive oil instead of butter'].map((t) => (
        <li key={t}>
          <CheckRoundedIcon /> {t}
        </li>
      ))}
    </ul>
    <div className="lp-plan__avoid" aria-hidden>
      <span className="lp-plan__avoid-label">
        <BlockRoundedIcon /> Limit
      </span>
      {['Sugary drinks', 'White bread', 'Processed meat'].map((t) => (
        <span key={t} className="lp-plan__chip">
          {t}
        </span>
      ))}
    </div>
  </div>
);

const TREND = [
  { m: 'Jan', p: 68 },
  { m: 'Feb', p: 64 },
  { m: 'Mar', p: 57 },
  { m: 'Apr', p: 52 },
  { m: 'May', p: 46 },
  { m: 'Jun', p: 41 },
];

// Self-contained SVG chart: scales by viewBox, no layout measurement needed.
const CH = { w: 300, h: 120, min: 30, max: 75 };
const tx = (i: number) => (i / (TREND.length - 1)) * CH.w;
const ty = (p: number) => CH.h - ((p - CH.min) / (CH.max - CH.min)) * CH.h;
const LINE = TREND.map((d, i) => `${i ? 'L' : 'M'}${tx(i).toFixed(1)} ${ty(d.p).toFixed(1)}`).join(' ');
const AREA = `${LINE} L${CH.w} ${CH.h} L0 ${CH.h} Z`;

const TrendSnippet = () => (
  <div className="lp-snippet lp-trend" role="img" aria-label="Sample progress chart: MetS probability falling from 68% in January to 41% in June.">
    <div className="lp-trend__head" aria-hidden>
      <div>
        <span className="lp-trend__label">MetS probability</span>
        <span className="lp-trend__value">41%</span>
      </div>
      <span className="lp-trend__delta">−27 pts in 6 months</span>
    </div>
    <div className="lp-trend__chart" aria-hidden>
      <div className="lp-trend__plot">
      <svg viewBox={`0 0 ${CH.w} ${CH.h}`} preserveAspectRatio="none" className="lp-trend__svg">
        <defs>
          <linearGradient id="lpTrendFill" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0F9D8F" stopOpacity="0.25" />
            <stop offset="100%" stopColor="#0F9D8F" stopOpacity="0" />
          </linearGradient>
        </defs>
        {[0.25, 0.5, 0.75].map((f) => (
          <line key={f} x1="0" x2={CH.w} y1={CH.h * f} y2={CH.h * f} className="lp-trend__grid" vectorEffect="non-scaling-stroke" />
        ))}
        <path d={AREA} fill="url(#lpTrendFill)" />
        <path d={LINE} fill="none" stroke="#0F9D8F" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" vectorEffect="non-scaling-stroke" />
      </svg>
      {/* HTML dot so it stays round when the SVG stretches */}
      <span
        className="lp-trend__dot"
        style={{ left: '100%', top: `${(ty(TREND[TREND.length - 1].p) / CH.h) * 100}%` }}
      />
      </div>
      <div className="lp-trend__x">
        {TREND.map((d) => (
          <span key={d.m}>{d.m}</span>
        ))}
      </div>
    </div>
  </div>
);

/* ── Rows ─────────────────────────────────────────────────────────────── */

interface Row {
  kicker: string;
  title: ReactNode;
  text: string;
  points: string[];
  snippet: ReactNode;
}

const ROWS: Row[] = [
  {
    kicker: 'Prediction',
    title: (
      <>
        Know where you stand. <em className="lp-serif">Clearly.</em>
      </>
    ),
    text: 'Enter a few numbers from a routine check-up and see how likely metabolic syndrome is, and how severe — so the result reflects both likelihood and degree.',
    points: ['Probability, not a yes/no label', 'Severity shown as low, moderate or high', 'Transparent marker-by-marker breakdown'],
    snippet: <PipelineSnippet />,
  },
  {
    kicker: 'Personal plan',
    title: (
      <>
        Advice shaped by <em className="lp-serif">your numbers.</em>
      </>
    ),
    text: 'Recommendations change with your severity grade — what to eat, what to limit, how to move and which yoga poses to try.',
    points: ['Diet plan and foods to limit', 'Exercise plan matched to your level', 'Yoga poses for stress and blood pressure'],
    snippet: <PlanSnippet />,
  },
  {
    kicker: 'Progress',
    title: (
      <>
        Watch your risk <em className="lp-serif">come down.</em>
      </>
    ),
    text: 'Save each assessment to your dashboard and see how lifestyle changes move your probability and every marker over time.',
    points: ['Assessment history in one place', 'Trends for risk and each marker', 'Free account — your data, your dashboard'],
    snippet: <TrendSnippet />,
  },
];

const FeatureRows = () => (
  <section id="features" className="lp-section" aria-label="Features">
    <Container maxWidth="lg">
      <div className="lp-rows">
        {ROWS.map((r, i) => (
          <div key={r.kicker} className={`lp-row${i % 2 ? ' is-flipped' : ''}`}>
            <AnimatedReveal className="lp-row__copy">
              <span className="lp-kicker">{r.kicker}</span>
              <Typography variant="h2" component="h2" className="lp-h2">
                {r.title}
              </Typography>
              <Typography variant="body1" className="lp-muted lp-row__text">
                {r.text}
              </Typography>
              <ul className="lp-checks">
                {r.points.map((p) => (
                  <li key={p}>
                    <CheckRoundedIcon aria-hidden />
                    {p}
                  </li>
                ))}
              </ul>
            </AnimatedReveal>
            <AnimatedReveal className="lp-row__visual" direction={i % 2 ? 'right' : 'left'}>
              {r.snippet}
            </AnimatedReveal>
          </div>
        ))}
      </div>
    </Container>
  </section>
);

export default FeatureRows;
