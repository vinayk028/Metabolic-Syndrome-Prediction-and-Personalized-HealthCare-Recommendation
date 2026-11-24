/**
 * FaqSection — two-column FAQ with a route to the AI assistant.
 * Anchored at #faq (the footer links to /about#faq).
 */
import { useState } from 'react';
import { Accordion, AccordionDetails, AccordionSummary, Button, Typography } from '@mui/material';
import AddRoundedIcon from '@mui/icons-material/AddRounded';
import ForumRoundedIcon from '@mui/icons-material/ForumRounded';
import { useChatStore } from '../../stores';
import { FAQS } from './content';

const FaqSection = () => {
  const [open, setOpen] = useState<number | false>(0);
  const { isOpen, toggleChat } = useChatStore();

  return (
    <section id="faq" className="ab-faq" aria-labelledby="ab-faq-title">
      <div className="ab-faq__aside">
        <span className="ab-eyebrow">FAQ</span>
        <Typography id="ab-faq-title" variant="h2" className="ab-h2">
          Questions, <em className="ab-serif">answered</em>
        </Typography>
        <p className="ab-lead">
          Can't find what you're looking for? Our AI assistant can explain metabolic syndrome, your markers and your
          results in plain language.
        </p>
        <Button variant="outlined" size="large" startIcon={<ForumRoundedIcon />} onClick={() => !isOpen && toggleChat()} sx={{ mt: 3 }}>
          Ask the assistant
        </Button>
      </div>

      <div className="ab-faq__list">
        {FAQS.map(({ q, a }, i) => (
          <Accordion
            key={q}
            disableGutters
            elevation={0}
            square
            expanded={open === i}
            onChange={(_, isExpanded) => setOpen(isExpanded ? i : false)}
            className="ab-faq__item"
          >
            <AccordionSummary
              expandIcon={<AddRoundedIcon />}
              id={`ab-faq-${i}-header`}
              aria-controls={`ab-faq-${i}-content`}
              className="ab-faq__q"
            >
              <Typography component="h3" className="ab-faq__q-text">
                {q}
              </Typography>
            </AccordionSummary>
            <AccordionDetails id={`ab-faq-${i}-content`} className="ab-faq__a">
              <Typography variant="body1">{a}</Typography>
            </AccordionDetails>
          </Accordion>
        ))}
      </div>
    </section>
  );
};

export default FaqSection;
