/**
 * AuthShell — shared layout for Sign in / Create account.
 * Wide screens: tinted brand panel (why an account helps) beside the form.
 * Narrow screens: just the form, with the brand mark above it.
 */
import type { ReactNode } from 'react';
import { Typography } from '@mui/material';
import CheckRoundedIcon from '@mui/icons-material/CheckRounded';
import { BrandMark, MetabolicRing } from '../../components/ui';
import './Auth.css';

const BENEFITS = [
  'Save every assessment automatically',
  'Track each marker and your risk over time',
  'Get reminders and a personal plan you can download',
];

interface AuthShellProps {
  eyebrow: string;
  title: ReactNode;
  subtitle: string;
  children: ReactNode;
  footer: ReactNode;
}

const AuthShell = ({ eyebrow, title, subtitle, children, footer }: AuthShellProps) => (
  <div className="au">
    <aside className="au-panel" aria-label="Why create an account">
      <div className="au-panel__copy">
        <span className="au-eyebrow">MetS Health account</span>
        <p className="au-panel__title">
          Your metabolic health, <em className="au-serif">in balance.</em>
        </p>
        <ul className="au-benefits">
          {BENEFITS.map((b) => (
            <li key={b}>
              <span className="au-benefits__tick">
                <CheckRoundedIcon />
              </span>
              {b}
            </li>
          ))}
        </ul>
      </div>
      <div className="au-panel__ring" aria-hidden>
        <MetabolicRing colorBy="factor" size={220} thickness={12} animate={false}>
          <span className="au-ring__n">5</span>
          <span className="au-ring__label">markers, one view</span>
        </MetabolicRing>
      </div>
    </aside>

    <section className="au-form-wrap">
      <div className="au-card">
        <header className="au-head">
          <span className="au-head__mark">
            <BrandMark size={44} />
          </span>
          <span className="au-eyebrow">{eyebrow}</span>
          <Typography variant="h2" component="h1" className="au-title">
            {title}
          </Typography>
          <Typography className="au-sub">{subtitle}</Typography>
        </header>
        {children}
        <footer className="au-footer">{footer}</footer>
      </div>
    </section>
  </div>
);

export default AuthShell;
