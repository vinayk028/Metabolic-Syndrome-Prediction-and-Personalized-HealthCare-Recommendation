/**
 * HowItWorks — numbered timeline (horizontal when there's room, vertical when not).
 */
import { Link } from 'react-router-dom';
import { Button, Container, Typography } from '@mui/material';
import ArrowForwardRoundedIcon from '@mui/icons-material/ArrowForwardRounded';
import { AnimatedReveal } from '../../../components/ui';
import { STEPS } from '../content';

const HowItWorks = () => (
  <section id="how-it-works" className="lp-section lp-section--sunken" aria-labelledby="lp-how-title">
    <Container maxWidth="lg">
      <div className="lp-head-row">
        <div>
          <span className="lp-kicker">How it works</span>
          <Typography id="lp-how-title" variant="h2" className="lp-h2">
            From check-up numbers to a <em className="lp-serif">clear plan.</em>
          </Typography>
        </div>
        <Button component={Link} to="/assessment" variant="gradient" size="large" endIcon={<ArrowForwardRoundedIcon />}>
          Start now
        </Button>
      </div>

      <ol className="lp-timeline">
        {STEPS.map(({ title, text }, i) => (
          <AnimatedReveal as="li" key={title} delay={i * 0.1} className="lp-timeline__item">
            <span className="lp-timeline__num" aria-hidden>
              {i + 1}
            </span>
            <Typography variant="h5" component="h3" className="lp-timeline__title">
              <span className="visually-hidden">Step {i + 1}: </span>
              {title}
            </Typography>
            <Typography variant="body2" className="lp-timeline__text">
              {text}
            </Typography>
          </AnimatedReveal>
        ))}
      </ol>
    </Container>
  </section>
);

export default HowItWorks;
