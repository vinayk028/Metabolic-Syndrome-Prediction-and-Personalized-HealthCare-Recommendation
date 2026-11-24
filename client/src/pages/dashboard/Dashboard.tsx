/**
 * Dashboard — personal overview of saved assessments: latest result, trend,
 * severity mix, latest markers, plan summary and history.
 *
 * Data flow is unchanged: assessment history comes from getAssessmentHistory()
 * and unauthenticated users are redirected to /login.
 */
import { useEffect, useMemo, useState, type ReactNode } from 'react';
import { useNavigate } from 'react-router-dom';
import { Button, IconButton, Tooltip, Typography } from '@mui/material';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import ShowChartRoundedIcon from '@mui/icons-material/ShowChartRounded';
import SignalCellularAltRoundedIcon from '@mui/icons-material/SignalCellularAltRounded';
import HistoryRoundedIcon from '@mui/icons-material/HistoryRounded';
import EventRoundedIcon from '@mui/icons-material/EventRounded';
import TrendingDownRoundedIcon from '@mui/icons-material/TrendingDownRounded';
import TrendingUpRoundedIcon from '@mui/icons-material/TrendingUpRounded';
import TrendingFlatRoundedIcon from '@mui/icons-material/TrendingFlatRounded';
import RestaurantRoundedIcon from '@mui/icons-material/RestaurantRounded';
import BlockRoundedIcon from '@mui/icons-material/BlockRounded';
import DirectionsRunRoundedIcon from '@mui/icons-material/DirectionsRunRounded';
import SelfImprovementRoundedIcon from '@mui/icons-material/SelfImprovementRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import TipsAndUpdatesRoundedIcon from '@mui/icons-material/TipsAndUpdatesRounded';
import CheckCircleRoundedIcon from '@mui/icons-material/CheckCircleRounded';
import EventRepeatRoundedIcon from '@mui/icons-material/EventRepeatRounded';
import {
  Area,
  AreaChart,
  CartesianGrid,
  ResponsiveContainer,
  Tooltip as RechartsTooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { useAuthStore } from '../../stores';
import { getAssessmentHistory } from '../../services/api';
import type { AssessmentHistoryItem } from '../../types/types';
import { EmptyState, PulseLoader, RangeBar, RiskBadge } from '../../components/ui';
import { FACTORS, toRiskKey } from '../../theme/factors';
import { STATUS_LABEL, buildMarkerRows } from '../../theme/markers';
import { brand, riskColors, type RiskKey } from '../../theme/tokens';
import { factorVar } from '../../theme/cssVars';
import { FOCUS_TIPS, RECHECK_DAYS, buildMarkerSeries, nextCheck } from './insights';
import './Dashboard.css';

const MAX_SAVED = 7;
const LEVEL_LABEL: Record<RiskKey, string> = { low: 'Low', medium: 'Moderate', high: 'High' };
const LEVEL_ORDER: RiskKey[] = ['low', 'medium', 'high'];

const fmtLong = (d: string) => new Date(d).toLocaleDateString(undefined, { year: 'numeric', month: 'long', day: 'numeric' });
const fmtShort = (d: string) => new Date(d).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
const pct = (p: number) => `${(p * 100).toFixed(1)}%`;

const daysAgo = (d: string) => {
  const days = Math.floor((Date.now() - new Date(d).getTime()) / 86_400_000);
  if (days <= 0) return 'Today';
  if (days === 1) return 'Yesterday';
  return `${days} days ago`;
};

const levelOf = (a: AssessmentHistoryItem): RiskKey => toRiskKey(a.riskLevel) ?? 'low';

/* ── Pieces ─────────────────────────────────────────────────────────────── */

const Card = ({
  title,
  icon,
  meta,
  action,
  className = '',
  children,
  id,
}: {
  title: string;
  icon: ReactNode;
  meta?: ReactNode;
  action?: ReactNode;
  className?: string;
  children: ReactNode;
  id: string;
}) => (
  <section className={`db-card ${className}`} aria-labelledby={id}>
    <header className="db-card__head">
      <span className="db-card__icon" aria-hidden>
        {icon}
      </span>
      <div className="db-card__titles">
        <Typography id={id} component="h2" className="db-card__title">
          {title}
        </Typography>
        {meta && <p className="db-card__meta">{meta}</p>}
      </div>
      {action}
    </header>
    {children}
  </section>
);

const Kpi = ({ icon, label, value, caption }: { icon: ReactNode; label: string; value: ReactNode; caption?: ReactNode }) => (
  <li className="db-kpi">
    <span className="db-kpi__icon" aria-hidden>
      {icon}
    </span>
    <span className="db-kpi__label">{label}</span>
    <span className="db-kpi__value">{value}</span>
    {caption && <span className="db-kpi__caption">{caption}</span>}
  </li>
);

/** Change in probability vs the previous assessment — down is good. */
const Delta = ({ value }: { value: number }) => {
  const pts = Math.abs(value * 100).toFixed(1);
  if (value < -0.001)
    return (
      <span className="db-delta is-good">
        <TrendingDownRoundedIcon aria-hidden /> {pts} pts lower than last time
      </span>
    );
  if (value > 0.001)
    return (
      <span className="db-delta is-bad">
        <TrendingUpRoundedIcon aria-hidden /> {pts} pts higher than last time
      </span>
    );
  return (
    <span className="db-delta">
      <TrendingFlatRoundedIcon aria-hidden /> No change since last time
    </span>
  );
};

interface TrendPoint {
  date: string;
  full: string;
  probability: number;
  level: RiskKey;
}

/** Recharts injects active/payload into the content element. */
const TrendTooltip = ({ active, payload }: { active?: boolean; payload?: { payload: TrendPoint }[] }) => {
  if (!active || !payload?.length) return null;
  const p = payload[0].payload;
  return (
    <div className="db-tip">
      <span className="db-tip__date">{p.full}</span>
      <span className="db-tip__value">{p.probability}%</span>
      <RiskBadge level={p.level} label={`${LEVEL_LABEL[p.level]} severity`} size="small" />
    </div>
  );
};

/** Tiny fixed-size line of a marker's history; last point emphasised. */
const Sparkline = ({ values, improving }: { values: number[]; improving: boolean | null }) => {
  const W = 88;
  const H = 28;
  const P = 4;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = max - min || 1;
  const pts = values.map((v, i) => [P + (i / (values.length - 1)) * (W - 2 * P), H - P - ((v - min) / span) * (H - 2 * P)]);
  const color = improving === true ? riskColors.low.main : improving === false ? riskColors.high.main : brand.slate[500];
  const [lx, ly] = pts[pts.length - 1];
  return (
    <svg className="db-spark" width={W} height={H} viewBox={`0 0 ${W} ${H}`} aria-hidden>
      <polyline
        points={pts.map((pt) => pt.join(',')).join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle className="db-dot" cx={lx} cy={ly} r={3.5} fill={color} stroke="#fff" strokeWidth={1.5} />
    </svg>
  );
};

/* ── Page ───────────────────────────────────────────────────────────────── */

const Dashboard = () => {
  const navigate = useNavigate();
  const { user, isAuthenticated, isLoading: authLoading } = useAuthStore();
  const [assessments, setAssessments] = useState<AssessmentHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);
  // Refresh happens in place (no full-page loader) — the icon spins instead.
  const [refreshing, setRefreshing] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    if (!authLoading && !isAuthenticated) {
      navigate('/login');
    }
  }, [authLoading, isAuthenticated, navigate]);

  useEffect(() => {
    const fetchHistory = async () => {
      if (!user) return;
      setLoading(true);
      try {
        const response = await getAssessmentHistory();
        if (response.success) {
          setAssessments(response.assessmentHistory || []);
          setUpdatedAt(new Date());
        }
      } catch (err) {
        console.error('Error fetching assessment history:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchHistory();
  }, [user]);

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    try {
      // Keep the spinner up long enough to read as feedback, not a flicker
      const [response] = await Promise.all([getAssessmentHistory(), new Promise((r) => setTimeout(r, 700))]);
      if (response.success) {
        setAssessments(response.assessmentHistory || []);
        setUpdatedAt(new Date());
      }
    } catch (err) {
      console.error('Error refreshing:', err);
    } finally {
      setRefreshing(false);
    }
  };

  // Oldest first (charts); newest is last
  const sorted = useMemo(
    () => [...assessments].sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()),
    [assessments],
  );
  const latest = sorted.length > 0 ? sorted[sorted.length - 1] : null;
  const previous = sorted.length > 1 ? sorted[sorted.length - 2] : null;

  const trendData = useMemo<TrendPoint[]>(
    () =>
      sorted.map((a) => ({
        date: fmtShort(a.date),
        full: fmtLong(a.date),
        probability: +(a.probability * 100).toFixed(1),
        level: levelOf(a),
      })),
    [sorted],
  );

  const levelCounts = useMemo(() => {
    const c: Record<RiskKey, number> = { low: 0, medium: 0, high: 0 };
    sorted.forEach((a) => c[levelOf(a)]++);
    return c;
  }, [sorted]);

  const markerRows = useMemo(() => {
    const p = latest?.inputParameters;
    if (!p) return [];
    return buildMarkerRows({
      gender: p.gender,
      systolicBP: p.systolicBP,
      diastolicBP: p.diastolicBP,
      waistCircumference: p.waistCircumference,
      hdlCholesterol: p.hdlCholesterol,
      triglyceride: p.triglyceride,
      fpg: p.fpg,
    });
  }, [latest]);

  const series = useMemo(() => buildMarkerSeries(sorted), [sorted]);
  // Out-of-range first, then borderline
  const focus = useMemo(
    () =>
      markerRows
        .filter((m) => m.status !== 'normal')
        .sort((a, b) => (a.status === b.status ? 0 : a.status === 'elevated' ? -1 : 1)),
    [markerRows],
  );

  /* ── Loading / auth ──────────────────────────────────────────────────── */

  if (authLoading || !user) return <PulseLoader label="Loading your dashboard…" fullPage />;
  if (loading) return <PulseLoader label="Fetching your health data…" fullPage />;

  const latestLevel = latest ? levelOf(latest) : null;
  const prevLevel = previous ? levelOf(previous) : null;
  const levelChange =
    latestLevel && prevLevel
      ? latestLevel === prevLevel
        ? 'Same as last time'
        : LEVEL_ORDER.indexOf(latestLevel) < LEVEL_ORDER.indexOf(prevLevel)
          ? `Improved from ${LEVEL_LABEL[prevLevel]}`
          : `Up from ${LEVEL_LABEL[prevLevel]}`
      : 'From your latest assessment';

  const recs = latest?.recommendations;
  const planTiles = recs
    ? [
        { label: 'Diet', icon: <RestaurantRoundedIcon />, n: recs.dietPlan?.length || 0 },
        { label: 'Foods to limit', icon: <BlockRoundedIcon />, n: recs.avoidList?.length || 0 },
        { label: 'Exercise', icon: <DirectionsRunRoundedIcon />, n: recs.exercisePlan?.length || 0 },
        { label: 'Yoga', icon: <SelfImprovementRoundedIcon />, n: recs.yogaPoses?.length || 0 },
      ]
    : [];

  const diagnoses = latest?.inputParameters
    ? [
        { label: 'Fatty liver', on: latest.inputParameters.fattyLiver },
        { label: 'Hypertension', on: latest.inputParameters.hypertension },
        { label: 'Diabetes', on: latest.inputParameters.diabetes },
      ]
    : [];

  return (
    <div className="db">
      {/* ── Header ─────────────────────────────────────────────────────── */}
      <header className="db-head">
        <div className="db-head__copy">
          <span className="db-eyebrow">Your dashboard</span>
          <Typography variant="h2" component="h1" className="db-head__title">
            Welcome back, <em className="db-serif">{user.firstName}</em>
          </Typography>
          <Typography className="db-head__sub">Here's how your metabolic health is trending.</Typography>
          {updatedAt && (
            <p className="db-head__updated" aria-live="polite">
              {refreshing
                ? 'Updating…'
                : `Updated ${updatedAt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`}
            </p>
          )}
        </div>
        <div className="db-head__actions">
          <Tooltip title={refreshing ? 'Refreshing…' : 'Refresh'}>
            <IconButton
              onClick={handleRefresh}
              aria-label="Refresh data"
              aria-busy={refreshing}
              className={`db-refresh${refreshing ? ' is-spinning' : ''}`}
            >
              <RefreshRoundedIcon />
            </IconButton>
          </Tooltip>
          <Button variant="gradient" size="large" startIcon={<AddRoundedIcon />} onClick={() => navigate('/assessment')}>
            New assessment
          </Button>
        </div>
      </header>

      {!latest ? (
        <EmptyState
          icon={<MonitorHeartRoundedIcon />}
          title="No assessments yet"
          description="Take your first assessment to unlock your dashboard — your trend, markers and personal plan will appear here."
          action={
            <Button variant="gradient" size="large" endIcon={<ArrowForwardRoundedIcon />} onClick={() => navigate('/assessment')}>
              Start your first assessment
            </Button>
          }
        />
      ) : (
        <>
          {/* ── Next check ────────────────────────────────────────────── */}
          {(() => {
            const { due, daysLeft } = nextCheck(latest.date);
            const overdue = daysLeft <= 0;
            return (
              <section className={`db-next${overdue ? ' is-due' : ''}`} aria-label="Next check">
                <span className="db-next__icon" aria-hidden>
                  <EventRepeatRoundedIcon />
                </span>
                <p className="db-next__text">
                  {overdue ? (
                    <>
                      <strong>Your next check is due.</strong> Your last assessment was {daysAgo(latest.date).toLowerCase()} — a
                      fresh one shows whether your changes are working.
                    </>
                  ) : (
                    <>
                      <strong>
                        Next check suggested around {due.toLocaleDateString(undefined, { day: 'numeric', month: 'long' })}
                      </strong>{' '}
                      (in {daysLeft} day{daysLeft === 1 ? '' : 's'}). Checking every {RECHECK_DAYS / 30} months shows
                      whether your changes are working.
                    </>
                  )}
                </p>
                <Button
                  variant={overdue ? 'gradient' : 'outlined'}
                  endIcon={<ArrowForwardRoundedIcon />}
                  onClick={() => navigate('/assessment')}
                  className="db-next__btn"
                >
                  {overdue ? 'Start check' : 'Check now'}
                </Button>
              </section>
            );
          })()}

          {/* ── KPIs ──────────────────────────────────────────────────── */}
          <ul className="db-kpis" aria-label="Summary">
            <Kpi
              icon={<MonitorHeartRoundedIcon />}
              label="Chance of MetS"
              value={pct(latest.probability)}
              caption={previous ? <Delta value={latest.probability - previous.probability} /> : 'Your first result'}
            />
            <Kpi
              icon={<SignalCellularAltRoundedIcon />}
              label="Severity"
              value={<RiskBadge level={latestLevel} label={LEVEL_LABEL[latestLevel!]} />}
              caption={levelChange}
            />
            <Kpi
              icon={<HistoryRoundedIcon />}
              label="Assessments"
              value={
                <>
                  {sorted.length}
                  <small> / {MAX_SAVED}</small>
                </>
              }
              caption={`Your latest ${MAX_SAVED} are kept`}
            />
            <Kpi icon={<EventRoundedIcon />} label="Last check" value={fmtShort(latest.date)} caption={daysAgo(latest.date)} />
          </ul>

          {/* ── Trend + severity mix ──────────────────────────────────── */}
          <div className="db-row db-row--trend">
            <Card
              id="db-trend"
              title="Chance of metabolic syndrome"
              icon={<ShowChartRoundedIcon />}
              meta={sorted.length > 1 ? `Across your last ${sorted.length} assessments` : 'Take another assessment to see your trend'}
            >
              <div className="db-chart" role="img" aria-label={`Trend: ${trendData.map((d) => `${d.date} ${d.probability}%`).join(', ')}`}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={trendData} margin={{ top: 12, right: 12, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="dbTrendFill" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor={brand.teal[500]} stopOpacity={0.22} />
                        <stop offset="100%" stopColor={brand.teal[500]} stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid vertical={false} stroke={brand.slate[200]} strokeDasharray="3 4" />
                    <XAxis dataKey="date" tickLine={false} axisLine={false} fontSize={12} stroke={brand.slate[500]} dy={6} />
                    <YAxis
                      domain={[0, 100]}
                      ticks={[0, 25, 50, 75, 100]}
                      unit="%"
                      tickLine={false}
                      axisLine={false}
                      fontSize={12}
                      stroke={brand.slate[500]}
                      width={44}
                    />
                    <RechartsTooltip
                      content={<TrendTooltip />}
                      cursor={{ stroke: brand.slate[300], strokeWidth: 1, strokeDasharray: '3 3' }}
                    />
                    <Area
                      type="monotone"
                      dataKey="probability"
                      name="Chance of MetS"
                      stroke={brand.teal[700]}
                      strokeWidth={2}
                      fill="url(#dbTrendFill)"
                      isAnimationActive={false}
                      dot={(props: { cx?: number; cy?: number; index?: number }) => (
                        <circle
                          key={props.index}
                          cx={props.cx}
                          cy={props.cy}
                          r={5}
                          className="db-dot"
                          fill={riskColors[trendData[props.index ?? 0]?.level ?? 'low'].main}
                          stroke="#fff"
                          strokeWidth={2}
                        />
                      )}
                      activeDot={{ r: 7, stroke: '#fff', strokeWidth: 2, fill: brand.teal[700] }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
              <ul className="db-legend" aria-label="Dot colour shows severity">
                {LEVEL_ORDER.map((k) => (
                  <li key={k}>
                    <span className="db-legend__dot" style={{ background: riskColors[k].main }} aria-hidden />
                    {LEVEL_LABEL[k]} severity
                  </li>
                ))}
              </ul>
            </Card>

            <Card id="db-mix" title="Severity mix" icon={<SignalCellularAltRoundedIcon />} meta={`Across ${sorted.length} assessment${sorted.length > 1 ? 's' : ''}`}>
              <ul className="db-mix">
                {LEVEL_ORDER.map((k) => {
                  const n = levelCounts[k];
                  const share = sorted.length ? (n / sorted.length) * 100 : 0;
                  return (
                    <li key={k} className="db-mix__row">
                      <div className="db-mix__top">
                        <RiskBadge level={k} label={LEVEL_LABEL[k]} size="small" />
                        <span className="db-mix__count">
                          {n} <small>{n === 1 ? 'time' : 'times'}</small>
                        </span>
                      </div>
                      <div className="db-mix__track" aria-hidden>
                        <span style={{ width: `${share}%`, background: riskColors[k].main }} />
                      </div>
                    </li>
                  );
                })}
              </ul>
            </Card>
          </div>

          {/* ── Your markers (latest + trend) ─────────────────────────── */}
          <Card
            id="db-markers"
            title="Your markers"
            icon={<MonitorHeartRoundedIcon />}
            meta={
              latest.inputParameters
                ? `Latest ${fmtLong(latest.date)} · ${latest.inputParameters.gender === 'Women' ? 'Female' : 'Male'}, ${latest.inputParameters.age} years`
                : fmtLong(latest.date)
            }
          >
            {markerRows.length > 0 ? (
              <ul className="db-markers">
                {markerRows.map((m) => {
                  const F = FACTORS[m.key];
                  const Icon = F.icon;
                  const s = series[m.key];
                  const first = s?.points[0];
                  return (
                    <li key={m.key} className="db-marker" style={{ ['--f' as string]: factorVar(m.key) }}>
                      <span className="db-marker__icon" aria-hidden>
                        <Icon />
                      </span>
                      <div className="db-marker__body">
                        <div className="db-marker__row">
                          <span className="db-marker__name">{F.label}</span>
                          <span className="db-marker__value">
                            {m.display} <small>{m.unit}</small>
                          </span>
                        </div>
                        <RangeBar
                          min={m.min}
                          max={m.max}
                          zones={m.zones}
                          value={m.value}
                          thickness={6}
                          label={`${F.label} ${m.display} ${m.unit}, ${STATUS_LABEL[m.status]}`}
                        />
                      </div>
                      <div className="db-marker__status">
                        <RiskBadge level={m.risk} label={STATUS_LABEL[m.status]} size="small" />
                      </div>
                      <div className="db-marker__trend">
                        {s && s.points.length > 1 && first ? (
                          <>
                            <Sparkline values={s.points.map((p) => p.value)} improving={s.improving} />
                            <span
                              className={`db-change${s.improving === true ? ' is-good' : s.improving === false ? ' is-bad' : ''}`}
                              title={`${first.display} on ${fmtLong(first.date)} → ${m.display} now`}
                            >
                              <span className="db-change__value">
                                {s.delta === 0 ? (
                                  <>
                                    <TrendingFlatRoundedIcon aria-hidden /> No change
                                  </>
                                ) : (
                                  <>
                                    {s.delta < 0 ? <TrendingDownRoundedIcon aria-hidden /> : <TrendingUpRoundedIcon aria-hidden />}
                                    {Math.abs(s.delta)} {s.unit}
                                  </>
                                )}
                              </span>
                              <small>since {fmtShort(first.date)}</small>
                            </span>
                          </>
                        ) : (
                          <span className="db-change db-change--na">Trend shows after your next check</span>
                        )}
                      </div>
                    </li>
                  );
                })}
              </ul>
            ) : (
              <p className="db-muted">Marker details are not available for this assessment.</p>
            )}

            {diagnoses.length > 0 && (
              <div className="db-diag" aria-label="Diagnoses">
                {diagnoses.map((d) => (
                  <span key={d.label} className={`db-diag__chip${d.on ? ' is-on' : ''}`}>
                    {d.label}: <strong>{d.on ? 'Yes' : 'No'}</strong>
                  </span>
                ))}
              </div>
            )}
          </Card>

          {/* ── Focus areas + plan ────────────────────────────────────── */}
          <div className="db-row db-row--details">
            <Card
              id="db-focus"
              title="Focus areas"
              icon={<TipsAndUpdatesRoundedIcon />}
              meta={focus.length ? 'Markers to work on, most important first' : 'From your latest assessment'}
            >
              {focus.length > 0 ? (
                <ul className="db-focus">
                  {focus.map((m) => {
                    const F = FACTORS[m.key];
                    const Icon = F.icon;
                    return (
                      <li key={m.key} className="db-focus__item" style={{ ['--f' as string]: factorVar(m.key) }}>
                        <span className="db-marker__icon" aria-hidden>
                          <Icon />
                        </span>
                        <div className="db-focus__body">
                          <div className="db-focus__head">
                            <span className="db-focus__name">{F.label}</span>
                            <RiskBadge level={m.risk} label={STATUS_LABEL[m.status]} size="small" />
                          </div>
                          <p className="db-focus__tip">{FOCUS_TIPS[m.key]}</p>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              ) : (
                <div className="db-allgood">
                  <CheckCircleRoundedIcon aria-hidden />
                  <p>
                    <strong>All your markers are in range.</strong> Keep up your current habits and recheck regularly.
                  </p>
                </div>
              )}
              <p className="db-footnote">General guidance only — talk to your doctor before changing medication or treatment.</p>
            </Card>

            <Card
              id="db-plan"
              title="Your plan"
              icon={<RestaurantRoundedIcon />}
              meta="From your latest assessment"
            >
              {planTiles.length > 0 ? (
                <>
                  <ul className="db-plan">
                    {planTiles.map((t) => (
                      <li key={t.label} className="db-plan__tile">
                        <span className="db-plan__icon" aria-hidden>
                          {t.icon}
                        </span>
                        <span className="db-plan__n">{t.n}</span>
                        <span className="db-plan__label">{t.label}</span>
                      </li>
                    ))}
                  </ul>
                  {recs?.dietPlan?.length ? (
                    <div className="db-picks">
                      <span className="db-picks__label">Top picks</span>
                      <ul>
                        {recs.dietPlan.slice(0, 3).map((item, i) => (
                          <li key={i}>{item}</li>
                        ))}
                      </ul>
                    </div>
                  ) : null}
                </>
              ) : (
                <p className="db-muted">No recommendations saved for this assessment.</p>
              )}
            </Card>
          </div>

          {/* ── History ───────────────────────────────────────────────── */}
          <Card id="db-history" title="Assessment history" icon={<HistoryRoundedIcon />} meta={`Your last ${sorted.length}, newest first`}>
            <ol className="db-history">
              {[...sorted].reverse().map((a, i) => {
                const level = levelOf(a);
                return (
                  <li key={a._id || i} className="db-history__row">
                    <span className="db-history__dot" style={{ background: riskColors[level].main }} aria-hidden />
                    <div className="db-history__when">
                      <span className="db-history__date">{fmtLong(a.date)}</span>
                      {i === 0 && <span className="db-history__tag">Latest</span>}
                    </div>
                    <div className="db-history__prob">
                      <span className="db-history__pct">{pct(a.probability)}</span>
                      <span className="db-history__bar" aria-hidden>
                        <span style={{ width: `${Math.min(100, a.probability * 100)}%` }} />
                      </span>
                    </div>
                    <RiskBadge level={level} label={`${LEVEL_LABEL[level]} severity`} size="small" />
                  </li>
                );
              })}
            </ol>
          </Card>
        </>
      )}
    </div>
  );
};

export default Dashboard;
