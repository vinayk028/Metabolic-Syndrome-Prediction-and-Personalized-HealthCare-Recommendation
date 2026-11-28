/**
 * About — what MetS Health is, what metabolic syndrome is, the five markers
 * (same icons, colours and thresholds as the assessment and dashboard), how
 * to get your numbers, risk factors and prevention, your data, and the FAQ.
 */
import { Link } from 'react-router-dom';
import { Button, Typography } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import PriorityHighRoundedIcon from '@mui/icons-material/PriorityHighRounded';
import { AnimatedReveal } from '../../components/ui';
import { FACTORS, FACTOR_ORDER } from '../../theme/factors';
import { factorVar } from '../../theme/cssVars';
import ABOUT_HERO_IMAGE from '../../assets/about_hero.webp';
import DOCTOR_IMAGE from '../../assets/doctor.jpg';
import HEALTH_CHECK_IMAGE from '../../assets/health_check.jpg';
import { NUMBER_GUIDES, PREVENTION, PRIVACY_POINTS, RISK_FACTORS } from './content';
import FaqSection from './FaqSection';
import './About.css';

const About = () => (
  <div className="ab">
    {/* ── Page heading ─────────────────────────────────────────────────── */}
    <header className="ab-title">
      <Typography variant="h2" component="h1" className="ab-title__h1">
        About MetS Health
      </Typography>
      <Typography className="ab-title__sub">
        Understand your risk of metabolic syndrome from five routine check-up numbers — and what to do about it.
      </Typography>
    </header>

    {/* ── Intro panel ─────────────────────────────────────────────────── */}
    <section className="ab-head" aria-labelledby="ab-intro-title">
      <div className="ab-head__copy">
        <span className="ab-eyebrow">Our purpose</span>
        <Typography id="ab-intro-title" variant="h2" className="ab-head__title">
          Five markers. <em className="vb-accent-text">One clear picture.</em>
        </Typography>
        <Typography className="ab-head__sub">
          Metabolic syndrome affects millions of people, often without symptoms. MetS Health turns the numbers from a
          routine check-up into a clear risk estimate, a severity level and a personal plan — so you know where you
          stand and what to do next.
        </Typography>
      </div>
      <div className="ab-head__photo">
        <img src={ABOUT_HERO_IMAGE} alt="Medical team at work" loading="eager" />
      </div>
    </section>

    {/* ── What it is + the five markers ───────────────────────────────── */}
    <section className="ab-what" aria-labelledby="ab-what-title">
      <div className="ab-what__intro">
        <span className="ab-eyebrow">What it is</span>
        <Typography id="ab-what-title" variant="h2" className="ab-h2">
          A cluster, <em className="vb-accent-text">not a single disease.</em>
        </Typography>
        <p className="ab-lead">
          Metabolic syndrome is a group of related conditions that together raise your risk of cardiovascular disease,
          type 2 diabetes and stroke. They often appear together, and each one makes the others worse.
        </p>

        <div className="ab-rule" role="img" aria-label="Three or more of the five markers out of range indicates metabolic syndrome">
          <div className="ab-rule__dots" aria-hidden>
            {[0, 1, 2, 3, 4].map((i) => (
              <span key={i} className={i < 3 ? 'is-on' : undefined} />
            ))}
          </div>
          <div className="ab-rule__text" aria-hidden>
            <strong>3 of 5</strong>
            <span>markers out of range means metabolic syndrome</span>
          </div>
        </div>
      </div>

      <div className="ab-card" id="markers">
        <div className="ab-card__head">
          <Typography component="h3" className="ab-card__title">
            The five markers
          </Typography>
          <span className="ab-card__meta">Out of range when…</span>
        </div>
        <ol className="ab-markers">
          {FACTOR_ORDER.map((k) => {
            const F = FACTORS[k];
            const Icon = F.icon;
            return (
              <li key={k} className="ab-marker" style={{ ['--f' as string]: factorVar(k) }}>
                <span className="ab-marker__icon" aria-hidden>
                  <Icon />
                </span>
                <div className="ab-marker__body">
                  <div className="ab-marker__head">
                    <span className="ab-marker__name">{F.label}</span>
                    <span className="ab-marker__threshold">{F.threshold}</span>
                  </div>
                  <p className="ab-marker__desc">{F.description}</p>
                </div>
              </li>
            );
          })}
        </ol>
      </div>
    </section>

    {/* ── Getting your numbers ────────────────────────────────────────── */}
    <section className="ab-section" aria-labelledby="ab-numbers-title">
      <div className="ab-section__head">
        <span className="ab-eyebrow">Before you start</span>
        <Typography id="ab-numbers-title" variant="h2" className="ab-h2">
          Getting your <em className="vb-accent-text">numbers right</em>
        </Typography>
        <p className="ab-lead">Accurate inputs give an accurate result. Here's where each number comes from.</p>
      </div>
      <ul className="ab-guides">
        {NUMBER_GUIDES.map(({ icon: Icon, factor, title, gives, tips }) => (
          <AnimatedReveal as="li" key={title} className="ab-guide">
            <div
              className="ab-guide__head"
              style={{ ['--f' as string]: factor ? factorVar(factor) : 'var(--vb-primary)' }}
            >
              <span className="ab-guide__icon" aria-hidden>
                <Icon />
              </span>
              <div>
                <Typography component="h3" className="ab-guide__title">
                  {title}
                </Typography>
                <p className="ab-guide__gives">{gives}</p>
              </div>
            </div>
            <ul className="ab-guide__tips">
              {tips.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          </AnimatedReveal>
        ))}
      </ul>
    </section>

    {/* ── Risk factors & prevention ───────────────────────────────────── */}
    <section className="ab-lists" aria-label="Risk factors and prevention">
      <AnimatedReveal className="ab-list ab-list--risk">
        <div className="ab-list__photo">
          <img src={DOCTOR_IMAGE} alt="" loading="lazy" />
        </div>
        <div className="ab-list__body">
          <Typography component="h2" className="ab-list__title">
            What raises your risk
          </Typography>
          <p className="ab-list__sub">Some you can change, some you can't — knowing them helps.</p>
          <ul>
            {RISK_FACTORS.map((t) => (
              <li key={t}>
                <span className="ab-list__mark" aria-hidden>
                  <PriorityHighRoundedIcon />
                </span>
                {t}
              </li>
            ))}
          </ul>
        </div>
      </AnimatedReveal>

      <AnimatedReveal className="ab-list ab-list--prevent" delay={0.08}>
        <div className="ab-list__photo">
          <img src={HEALTH_CHECK_IMAGE} alt="" loading="lazy" />
        </div>
        <div className="ab-list__body">
          <Typography component="h2" className="ab-list__title">
            Prevention &amp; management
          </Typography>
          <p className="ab-list__sub">Small, steady changes make the biggest difference.</p>
          <ul>
            {PREVENTION.map((t) => (
              <li key={t}>
                <span className="ab-list__mark" aria-hidden>
                  <CheckRoundedIcon />
                </span>
                {t}
              </li>
            ))}
          </ul>
        </div>
      </AnimatedReveal>
    </section>

    {/* ── Your data & privacy ─────────────────────────────────────────── */}
    <section className="ab-section" aria-labelledby="ab-privacy-title">
      <div className="ab-section__head">
        <span className="ab-eyebrow">Your data</span>
        <Typography id="ab-privacy-title" variant="h2" className="ab-h2">
          Your data, <em className="vb-accent-text">your choice</em>
        </Typography>
      </div>
      <ul className="ab-privacy">
        {PRIVACY_POINTS.map(({ icon: Icon, title, text }) => (
          <li key={title}>
            <span className="ab-privacy__icon" aria-hidden>
              <Icon />
            </span>
            <Typography component="h3" className="ab-privacy__title">
              {title}
            </Typography>
            <p>{text}</p>
          </li>
        ))}
      </ul>
    </section>

    {/* ── FAQ ─────────────────────────────────────────────────────────── */}
    <FaqSection />

    {/* ── CTA ─────────────────────────────────────────────────────────── */}
    <section className="ab-cta" aria-labelledby="ab-cta-title">
      <div>
        <Typography id="ab-cta-title" component="h2" className="ab-cta__title">
          Know where you stand
        </Typography>
        <p className="ab-cta__text">Check your five markers in about two minutes — free, no sign-up needed.</p>
      </div>
      <Button component={Link} to="/assessment" variant="gradient" size="large" endIcon={<ArrowForwardRoundedIcon />}>
        Start free assessment
      </Button>
    </section>
  </div>
);

export default About;
