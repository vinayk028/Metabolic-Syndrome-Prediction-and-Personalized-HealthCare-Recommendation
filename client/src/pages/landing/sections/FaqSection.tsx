/**
 * FaqSection — editorial two-column FAQ with a route to the AI assistant.
 */
import { useState } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Button, Container, Typography } from '@mui/material';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import ForumRoundedIcon from '@mui/icons-material/ForumRounded';
import { useChatStore } from '../../../stores';
import { FAQS } from '../content';

const FaqSection = () => {
  const [open, setOpen] = useState<number | false>(0);
  const { isOpen, toggleChat } = useChatStore();

  return (
    <section id="faq" className="lp-section" aria-labelledby="lp-faq-title">
      <Container maxWidth="lg" className="lp-split">
        <div className="lp-split__aside">
          <span className="lp-kicker">FAQ</span>
          <Typography id="lp-faq-title" variant="h2" className="lp-h2">
            Questions, <em className="lp-serif">answered.</em>
          </Typography>
          <Typography variant="body1" className="lp-muted">
            Can't find what you're looking for? Our AI assistant can explain metabolic syndrome, your markers and your
            results in plain language.
          </Typography>
          <Button
            variant="outlined"
            size="large"
            startIcon={<ForumRoundedIcon />}
            onClick={() => !isOpen && toggleChat()}
            sx={{ mt: 3 }}
          >
            Ask the assistant
          </Button>
        </div>

        <div className="lp-faq">
          {FAQS.map(({ q, a }, i) => (
            <Accordion
              key={q}
              disableGutters
              elevation={0}
              square
              expanded={open === i}
              onChange={(_, isExpanded) => setOpen(isExpanded ? i : false)}
              className="lp-faq__item"
            >
              <AccordionSummary
                expandIcon={<AddRoundedIcon />}
                id={`lp-faq-${i}-header`}
                aria-controls={`lp-faq-${i}-content`}
                className="lp-faq__q"
              >
                <Typography component="h3" className="lp-faq__q-text">
                  {q}
                </Typography>
              </AccordionSummary>
              <AccordionDetails id={`lp-faq-${i}-content`} className="lp-faq__a">
                <Typography variant="body1">{a}</Typography>
              </AccordionDetails>
            </Accordion>
          ))}
        </div>
      </Container>
    </section>
  );
};

export default FaqSection;
