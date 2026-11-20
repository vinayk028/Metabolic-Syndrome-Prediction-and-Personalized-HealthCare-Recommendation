/**
 * WhatWeProvide — the six things the product gives you, right under the hero.
 */
import { Container, Typography } from '@mui/material';
import { AnimatedReveal } from '../../../components/ui';
import { FEATURES } from '../content';

const WhatWeProvide = () => (
  <section id="features-overview" className="lp-section" aria-labelledby="lp-provide-title">
    <Container maxWidth="lg">
      <div className="lp-head-row">
        <div>
          <span className="lp-kicker">What we provide</span>
          <Typography id="lp-provide-title" variant="h2" className="lp-h2">
            Everything you need, <em className="lp-serif">in one place.</em>
          </Typography>
        </div>
      </div>

      <ul className="lp-provide">
        {FEATURES.map(({ icon: Icon, title, text, from, to }, i) => (
          <AnimatedReveal
            as="li"
            key={title}
            delay={(i % 3) * 0.06}
            className="lp-provide__item"
            style={{ ['--ic1' as string]: from, ['--ic2' as string]: to }}
          >
            <span className="lp-provide__icon" aria-hidden>
              <Icon />
            </span>
            <Typography component="h3" className="lp-provide__title">
              {title}
            </Typography>
            <Typography variant="body2" className="lp-muted">
              {text}
            </Typography>
          </AnimatedReveal>
        ))}
      </ul>
    </Container>
  </section>
);

export default WhatWeProvide;
