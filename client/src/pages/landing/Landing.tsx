/**
 * Landing — public home page at "/" for every visitor (logged in or not).
 * Full-bleed route (see FULL_BLEED_ROUTES): each section owns its width.
 */
import HeroSection from './sections/HeroSection';
import WhatWeProvide from './sections/WhatWeProvide';
import HowItWorks from './sections/HowItWorks';
import FeatureRows from './sections/FeatureRows';
import NewsCarousel from './sections/NewsCarousel';
import './Landing.css';

const Landing = () => (
  <div className="lp">
    <HeroSection />
    <WhatWeProvide />
    <HowItWorks />
    <FeatureRows />
    <NewsCarousel />
  </div>
);

export default Landing;
