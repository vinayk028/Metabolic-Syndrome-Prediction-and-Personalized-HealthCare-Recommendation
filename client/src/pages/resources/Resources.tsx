/**
 * Resources — latest metabolic-health news and trusted external sources.
 * Data flow unchanged: news comes from useNewsStore().fetchResourceNews().
 * (The FAQ now lives on the About page.)
 */
import { useEffect, useState, type ReactNode, type SyntheticEvent } from 'react';
import { Button, Skeleton, Typography } from '@mui/material';
import RefreshRoundedIcon from '@mui/icons-material/RefreshRounded';
import OpenInNewRoundedIcon from '@mui/icons-material/OpenInNewRounded';
import NewspaperRoundedIcon from '@mui/icons-material/NewspaperRounded';
import LocalHospitalRoundedIcon from '@mui/icons-material/LocalHospitalRounded';
import ScienceRoundedIcon from '@mui/icons-material/ScienceRounded';
import DirectionsRunRoundedIcon from '@mui/icons-material/DirectionsRunRounded';
import MonitorHeartRoundedIcon from '@mui/icons-material/MonitorHeartRounded';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import { useNewsStore } from '../../stores';
import DEFAULT_NEWS_IMAGE from '../../assets/news.png';
import './Resources.css';

const getTimeAgo = (dateString: string): string => {
  const seconds = Math.floor((Date.now() - new Date(dateString).getTime()) / 1000);
  if (seconds < 60) return 'Just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d ago`;
  return new Date(dateString).toLocaleDateString(undefined, { day: 'numeric', month: 'short', year: 'numeric' });
};

interface Source {
  title: string;
  description: string;
  url: string;
}

interface SourceGroup {
  title: string;
  sub: string;
  icon: ReactNode;
  tone: string;
  items: Source[];
}

// Ordered so each row pairs groups of similar length (4+3, 2+2)
const SOURCE_GROUPS: SourceGroup[] = [
  {
    title: 'Medical organizations',
    sub: 'Clinical overviews from trusted health bodies',
    icon: <LocalHospitalRoundedIcon />,
    tone: '#e11d48',
    items: [
      {
        title: 'American Heart Association',
        description: 'Comprehensive information about metabolic syndrome and heart health',
        url: 'https://www.heart.org/en/health-topics/metabolic-syndrome',
      },
      {
        title: 'Mayo Clinic',
        description: 'Symptoms, causes and treatment of metabolic syndrome',
        url: 'https://www.mayoclinic.org/diseases-conditions/metabolic-syndrome/symptoms-causes/syc-20351916',
      },
      {
        title: 'National Heart, Lung, and Blood Institute',
        description: 'In-depth medical research and guidelines',
        url: 'https://www.nhlbi.nih.gov/health-topics/metabolic-syndrome',
      },
      {
        title: 'World Health Organization',
        description: 'Global health data on noncommunicable diseases',
        url: 'https://www.who.int/news-room/fact-sheets/detail/noncommunicable-diseases',
      },
    ],
  },
  {
    title: 'Lifestyle',
    sub: 'Practical guides for diet, activity and stress',
    icon: <DirectionsRunRoundedIcon />,
    tone: '#16a34a',
    items: [
      {
        title: 'DASH Diet',
        description: 'Dietary approaches to stop hypertension',
        url: 'https://www.nhlbi.nih.gov/health-topics/dash-eating-plan',
      },
      {
        title: 'Physical Activity Guidelines — CDC',
        description: 'Recommended physical activity for adults',
        url: 'https://www.cdc.gov/physicalactivity/basics/index.htm',
      },
      {
        title: 'Stress Management — Mayo Clinic',
        description: 'Techniques for managing stress effectively',
        url: 'https://www.mayoclinic.org/healthy-lifestyle/stress-management/in-depth/stress-management/art-20044289',
      },
    ],
  },
  {
    title: 'Research & studies',
    sub: 'Where the evidence comes from',
    icon: <ScienceRoundedIcon />,
    tone: '#2563eb',
    items: [
      {
        title: 'PubMed',
        description: 'Latest research articles on metabolic syndrome',
        url: 'https://pubmed.ncbi.nlm.nih.gov/?term=metabolic+syndrome',
      },
      {
        title: 'American Diabetes Association',
        description: 'Research on the link between metabolic syndrome and diabetes',
        url: 'https://diabetes.org/diabetes-risk/prediabetes/metabolic-syndrome',
      },
    ],
  },
  {
    title: 'Monitoring',
    sub: 'Understand and track your numbers at home',
    icon: <MonitorHeartRoundedIcon />,
    tone: '#7c3aed',
    items: [
      {
        title: 'Blood Pressure Readings — AHA',
        description: 'Guide to understanding blood pressure readings',
        url: 'https://www.heart.org/en/health-topics/high-blood-pressure/understanding-blood-pressure-readings',
      },
      {
        title: 'Blood Glucose Monitoring — CDC',
        description: 'How to monitor and manage blood sugar levels',
        url: 'https://www.cdc.gov/diabetes/managing/managing-blood-sugar/bloodglucosemonitoring.html',
      },
    ],
  },
];

const Resources = () => {
  const {
    resourceArticles: news,
    resourceLoading: newsLoading,
    resourceError: newsError,
    fetchResourceNews: fetchNews,
  } = useNewsStore();

  // Refresh happens in place: the current stories stay on screen and only the
  // icon spins (skeletons are for the very first load).
  const [refreshing, setRefreshing] = useState(false);
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null);

  useEffect(() => {
    fetchNews().then(() => setUpdatedAt(new Date()));
  }, [fetchNews]);

  const handleRefresh = async () => {
    if (refreshing) return;
    setRefreshing(true);
    // Keep the spinner up long enough to read as feedback, not a flicker
    await Promise.all([fetchNews(), new Promise((r) => setTimeout(r, 700))]);
    setUpdatedAt(new Date());
    setRefreshing(false);
  };

  const busy = refreshing || newsLoading;
  const firstLoad = newsLoading && news.length === 0;

  const handleImageError = (e: SyntheticEvent<HTMLImageElement>) => {
    e.currentTarget.src = DEFAULT_NEWS_IMAGE;
  };

  return (
    <div className="rs">
      {/* ── Page heading ─────────────────────────────────────────────────── */}
      <header className="rs-title">
        <Typography variant="h2" component="h1" className="rs-title__h1">
          Resources &amp; References
        </Typography>
        <Typography className="rs-title__sub">
          The latest metabolic-health news and trusted sources to learn more.
        </Typography>
      </header>

      {/* ── News ─────────────────────────────────────────────────────────── */}
      <section className="rs-section" aria-labelledby="rs-news-title" aria-busy={busy}>
        <div className="rs-section__head">
          <span className="rs-section__icon" aria-hidden>
            <NewspaperRoundedIcon />
          </span>
          <div className="rs-section__titles">
            <Typography id="rs-news-title" component="h2" className="rs-section__title">
              Latest news
            </Typography>
            <p className="rs-section__sub">
              Fresh stories on metabolic health from trusted outlets
              {updatedAt && (
                <span className="rs-updated" aria-live="polite">
                  {' · '}
                  {busy ? 'Updating…' : `Updated ${updatedAt.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' })}`}
                </span>
              )}
            </p>
          </div>
          <Button
            onClick={handleRefresh}
            disabled={busy}
            startIcon={<RefreshRoundedIcon className={busy ? 'rs-spin' : undefined} />}
            className="rs-refresh"
          >
            Refresh
          </Button>
        </div>

        {firstLoad ? (
          <ul className="rs-news" aria-hidden>
            {Array.from({ length: 6 }).map((_, i) => (
              <li key={i} className="rs-news__card">
                <Skeleton variant="rectangular" className="rs-news__img" />
                <div className="rs-news__body">
                  <Skeleton width="45%" />
                  <Skeleton width="95%" height={28} />
                  <Skeleton width="80%" />
                  <Skeleton width="60%" />
                </div>
              </li>
            ))}
          </ul>
        ) : newsError && news.length === 0 ? (
          <div className="rs-empty">
            <NewspaperRoundedIcon aria-hidden />
            <p>{newsError}</p>
            <Button variant="outlined" startIcon={<RefreshRoundedIcon className={busy ? 'rs-spin' : undefined} />} onClick={handleRefresh} disabled={busy}>
              Try again
            </Button>
          </div>
        ) : news.length === 0 ? (
          <div className="rs-empty">
            <NewspaperRoundedIcon aria-hidden />
            <p>No news right now — check back soon.</p>
          </div>
        ) : (
          <ul className="rs-news">
            {news.map((article, i) => (
              <li key={article.url || i} className="rs-news__card">
                <a href={article.url} target="_blank" rel="noopener noreferrer" className="rs-news__link">
                  <img
                    src={article.image || DEFAULT_NEWS_IMAGE}
                    alt=""
                    loading="lazy"
                    className="rs-news__img"
                    onError={handleImageError}
                  />
                  <div className="rs-news__body">
                    <span className="rs-news__meta">
                      <span className="rs-news__source">{article.source}</span>
                      <span aria-hidden>·</span>
                      <span>{getTimeAgo(article.publishedAt)}</span>
                    </span>
                    <Typography component="h3" className="rs-news__title">
                      {article.title}
                    </Typography>
                    {article.description && <p className="rs-news__desc">{article.description}</p>}
                    <span className="rs-news__read">
                      Read article <OpenInNewRoundedIcon aria-hidden />
                      <span className="visually-hidden"> (opens in a new tab)</span>
                    </span>
                  </div>
                </a>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* ── Trusted sources ──────────────────────────────────────────────── */}
      <section aria-labelledby="rs-sources-title">
        <div className="rs-sources__head">
          <Typography id="rs-sources-title" component="h2" className="rs-h2">
            Trusted sources
          </Typography>
          <p className="rs-section__sub">Hand-picked references if you want to go deeper.</p>
        </div>
        <div className="rs-groups">
          {SOURCE_GROUPS.map((g) => (
            <section key={g.title} className="rs-group" style={{ ['--t' as string]: g.tone }} aria-label={g.title}>
              <header className="rs-group__head">
                <span className="rs-group__icon" aria-hidden>
                  {g.icon}
                </span>
                <div>
                  <Typography component="h3" className="rs-group__title">
                    {g.title}
                  </Typography>
                  <p className="rs-group__sub">{g.sub}</p>
                </div>
              </header>
              <ul className="rs-links">
                {g.items.map((s) => (
                  <li key={s.url}>
                    <a href={s.url} target="_blank" rel="noopener noreferrer" className="rs-link">
                      <span className="rs-link__text">
                        <span className="rs-link__title">{s.title}</span>
                        <span className="rs-link__desc">{s.description}</span>
                      </span>
                      <OpenInNewRoundedIcon className="rs-link__icon" aria-hidden />
                      <span className="visually-hidden"> (opens in a new tab)</span>
                    </a>
                  </li>
                ))}
              </ul>
            </section>
          ))}
        </div>
      </section>

      {/* ── Disclaimer ───────────────────────────────────────────────────── */}
      <aside className="rs-note">
        <InfoOutlinedIcon aria-hidden />
        <p>
          <strong>For information only.</strong> MetS Health and the sources above don't replace professional medical
          advice, diagnosis or treatment. Always talk to a qualified healthcare provider about your health.
        </p>
      </aside>
    </div>
  );
};

export default Resources;
