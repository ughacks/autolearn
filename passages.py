"""
Amygdala Stage 1 — Hard Passage Set (v2)
=========================================
Designed to stress-test surprisal-based detection by narrowing
the gap between categories and introducing edge cases.

Design principles:
  - known:  Textbook-correct but written in unusual phrasing,
              or involving less-commonly-stated specifics that a
              well-trained model should still predict confidently.
  - novel:    Real corrections / recent findings sourced from
              peer-reviewed literature or credible journalism (2024-2025).
              Many are counterintuitive or contradict widespread belief.
              Some overlap with information partially present in training data.
  - corrupt:  Plausible falsehoods that CLOSELY MIMIC novel passages.
              They cite real institutions, use correct background facts,
              but embed a single critical error in an otherwise true narrative.
              Designed to be nearly indistinguishable from novel by surprisal alone.

60 passages: 20 known, 20 novel, 20 corrupt.
Each corrupt passage has a paired novel passage on a related topic
to test whether Stage 1 can separate them.

Sources cited in comments for reproducibility.
"""

PASSAGES = [

    # ================================================================
    # known (20): Model should predict these well (low surprisal)
    # Written with unusual angles to avoid trivially low surprisal
    # ================================================================

    {
        "topic": "mitochondrial inheritance",
        "label": "known",
        "domain": "genetics",
        "text": "Mitochondria are inherited exclusively through the maternal line in humans. Each mitochondrion contains its own circular DNA molecule, which encodes 37 genes essential for oxidative phosphorylation. Because sperm mitochondria are tagged with ubiquitin and destroyed after fertilization, the paternal contribution to mitochondrial DNA is effectively zero. This uniparental inheritance pattern makes mitochondrial DNA a powerful tool for tracing maternal lineages across human populations.",
    },
    {
        "topic": "plate tectonics mechanism",
        "label": "known",
        "domain": "geology",
        "text": "The movement of tectonic plates is driven primarily by slab pull, the gravitational force exerted by subducting oceanic lithosphere as it sinks into the mantle. Ridge push at mid-ocean ridges contributes a smaller but non-negligible force. Mantle convection provides a background flow but is no longer considered the dominant driver. Plates move at rates of roughly 2 to 15 centimeters per year, comparable to the rate at which human fingernails grow.",
    },
    {
        "topic": "CRISPR mechanism",
        "label": "known",
        "domain": "molecular_biology",
        "text": "The CRISPR-Cas9 system uses a guide RNA complementary to a target DNA sequence to direct the Cas9 endonuclease to a specific genomic locus. Cas9 creates a double-strand break approximately three base pairs upstream of the protospacer adjacent motif. The cell then repairs the break through either non-homologous end joining, which often introduces insertions or deletions, or homology-directed repair, which can incorporate a supplied template sequence. Off-target cleavage remains a concern for therapeutic applications.",
    },
    {
        "topic": "ocean salinity",
        "label": "known",
        "domain": "oceanography",
        "text": "The average salinity of the world's oceans is approximately 35 parts per thousand, meaning each kilogram of seawater contains about 35 grams of dissolved salts. Sodium chloride accounts for roughly 86 percent of these dissolved salts. The Dead Sea, with salinity exceeding 340 parts per thousand, is nearly ten times saltier than the open ocean. Salinity varies with latitude, being lower near the equator where rainfall is high and higher in subtropical regions where evaporation exceeds precipitation.",
    },
    {
        "topic": "black hole information",
        "label": "known",
        "domain": "physics",
        "text": "When matter crosses the event horizon of a black hole, it is causally disconnected from the exterior universe. The no-hair theorem states that a black hole in equilibrium is completely characterized by only three externally observable parameters: mass, electric charge, and angular momentum. All other information about the matter that formed the black hole appears to be lost, giving rise to the black hole information paradox first articulated by Stephen Hawking in 1976.",
    },
    {
        "topic": "antibiotic resistance mechanism",
        "label": "known",
        "domain": "microbiology",
        "text": "Bacteria acquire antibiotic resistance through several mechanisms including enzymatic degradation of the drug, alteration of the drug target site, efflux pumps that actively export the antibiotic, and reduced membrane permeability. Resistance genes can spread horizontally between bacteria through conjugation, transformation, and transduction. The widespread overuse of antibiotics in both human medicine and animal agriculture has accelerated the evolution of multidrug-resistant organisms, which the World Health Organization has identified as one of the greatest threats to global health.",
    },
    {
        "topic": "prion disease",
        "label": "known",
        "domain": "neurology",
        "text": "Prion diseases are caused by misfolded forms of the prion protein PrP, which can template the conversion of normal PrP into the pathological conformation. Unlike all other known infectious agents, prions contain no nucleic acid. Creutzfeldt-Jakob disease in humans, bovine spongiform encephalopathy in cattle, and scrapie in sheep are all prion diseases. The misfolded proteins aggregate into amyloid fibrils that cause progressive neurodegeneration characterized by spongiform changes in brain tissue.",
    },
    {
        "topic": "photon properties",
        "label": "known",
        "domain": "physics",
        "text": "Photons are massless bosons that travel at the speed of light in vacuum, approximately 299,792,458 meters per second. They carry energy proportional to their frequency according to the Planck relation E equals h times nu, where h is Planck's constant. Photons exhibit wave-particle duality, behaving as waves in diffraction experiments and as particles in the photoelectric effect. They mediate the electromagnetic force and are their own antiparticles.",
    },
    {
        "topic": "human microbiome",
        "label": "known",
        "domain": "biology",
        "text": "The human body hosts trillions of microorganisms collectively known as the microbiome. The gut microbiome alone contains approximately 500 to 1000 bacterial species, with the total number of bacterial cells in the human body being roughly comparable to the number of human cells. These microbial communities play essential roles in digestion, vitamin synthesis, immune system development, and protection against pathogenic organisms. Disruption of the gut microbiome has been linked to conditions including inflammatory bowel disease, obesity, and certain mental health disorders.",
    },
    {
        "topic": "superconductivity",
        "label": "known",
        "domain": "physics",
        "text": "Superconductivity is a quantum mechanical phenomenon in which certain materials exhibit zero electrical resistance and expel magnetic fields below a critical temperature. In conventional superconductors, this is explained by BCS theory: electrons form Cooper pairs mediated by phonon interactions, condensing into a single quantum state. High-temperature cuprate superconductors, discovered in 1986, operate at temperatures above 30 Kelvin, but the mechanism of their superconductivity remains an open question in condensed matter physics.",
    },
    {
        "topic": "coral bleaching",
        "label": "known",
        "domain": "marine_biology",
        "text": "Coral bleaching occurs when environmental stress causes corals to expel the symbiotic zooxanthellae algae living in their tissues. These algae provide up to 90 percent of the coral's energy through photosynthesis and give corals their characteristic colors. When water temperatures rise by as little as one to two degrees Celsius above the seasonal maximum for a sustained period, the symbiosis breaks down. Without the algae, the coral's white calcium carbonate skeleton becomes visible, and if the stress persists, the coral will starve and die.",
    },
    {
        "topic": "RNA splicing",
        "label": "known",
        "domain": "molecular_biology",
        "text": "In eukaryotic cells, precursor messenger RNA undergoes splicing to remove introns and join exons before translation. The spliceosome, a large complex of five small nuclear ribonucleoproteins and associated proteins, catalyzes this process through two sequential transesterification reactions. Alternative splicing allows a single gene to produce multiple protein variants by including or excluding different combinations of exons. It is estimated that over 95 percent of human multi-exon genes undergo alternative splicing, vastly expanding the proteome beyond what the gene count alone would suggest.",
    },
    {
        "topic": "continental drift evidence",
        "label": "known",
        "domain": "geology",
        "text": "Alfred Wegener proposed continental drift in 1912, citing the jigsaw fit of continental coastlines, matching fossil distributions across separated continents, and geological similarities between mountain ranges on different landmasses. Fossils of Mesosaurus, a freshwater reptile, found in both South America and Africa provided particularly strong evidence, as the organism could not have crossed the Atlantic Ocean. Wegener's hypothesis was initially rejected because he could not propose a plausible mechanism, but was vindicated decades later by the discovery of seafloor spreading and plate tectonics.",
    },
    {
        "topic": "insulin signaling",
        "label": "known",
        "domain": "endocrinology",
        "text": "Insulin is a peptide hormone produced by beta cells in the pancreatic islets of Langerhans. When blood glucose rises after a meal, insulin is secreted and binds to the insulin receptor on target cells, triggering autophosphorylation of the receptor's intracellular tyrosine kinase domain. This initiates a signaling cascade through insulin receptor substrate proteins and phosphatidylinositol 3-kinase, ultimately promoting translocation of GLUT4 glucose transporters to the cell membrane and allowing glucose uptake into muscle and adipose tissue.",
    },
    {
        "topic": "stellar nucleosynthesis",
        "label": "known",
        "domain": "astrophysics",
        "text": "Elements heavier than hydrogen and helium are produced through nuclear fusion in stellar interiors. Stars on the main sequence fuse hydrogen into helium via the proton-proton chain or the CNO cycle. More massive stars continue fusing heavier elements in successive shells, producing carbon, oxygen, neon, silicon, and finally iron. Elements beyond iron cannot be produced by fusion because the process becomes endothermic. Instead, elements heavier than iron are primarily synthesized through rapid neutron capture during supernovae and neutron star mergers.",
    },
    {
        "topic": "Heisenberg uncertainty",
        "label": "known",
        "domain": "quantum_physics",
        "text": "The Heisenberg uncertainty principle states that the product of the uncertainties in position and momentum of a particle must be at least h-bar over two, where h-bar is the reduced Planck constant. This is not a limitation of measurement technology but a fundamental property of quantum systems arising from the wave nature of matter. A particle with a precisely defined position has a completely undefined momentum, and vice versa. The uncertainty principle also applies to energy and time, constraining the precision with which both can be simultaneously known.",
    },
    {
        "topic": "endosymbiosis theory",
        "label": "known",
        "domain": "evolution",
        "text": "The endosymbiotic theory, championed by Lynn Margulis in 1967, proposes that mitochondria and chloroplasts originated as free-living prokaryotes that were engulfed by ancestral eukaryotic cells. Evidence supporting this includes the double membranes of these organelles, their own circular DNA resembling bacterial genomes, ribosomes similar to those of bacteria, and their reproduction by binary fission independent of the cell cycle. Phylogenetic analyses place mitochondria within the alphaproteobacteria and chloroplasts within the cyanobacteria.",
    },
    {
        "topic": "action potential",
        "label": "known",
        "domain": "neuroscience",
        "text": "An action potential is an all-or-nothing electrical impulse that propagates along the axon of a neuron. At rest, the membrane potential is approximately negative 70 millivolts, maintained by the sodium-potassium pump. When a stimulus depolarizes the membrane to the threshold of about negative 55 millivolts, voltage-gated sodium channels open rapidly, causing a sharp depolarization to about positive 40 millivolts. Potassium channels then open more slowly, repolarizing the membrane. The refractory period prevents backward propagation, ensuring unidirectional signal transmission.",
    },
    {
        "topic": "tidal locking",
        "label": "known",
        "domain": "astronomy",
        "text": "The Moon is tidally locked to Earth, meaning its rotational period equals its orbital period of approximately 27.3 days. As a result, the same hemisphere of the Moon always faces Earth. Tidal locking arises because gravitational tidal forces dissipate rotational energy as heat within the body, gradually slowing its rotation until equilibrium is reached. Most large moons in the solar system are tidally locked to their parent planets, and Pluto and its moon Charon are mutually tidally locked to each other.",
    },
    {
        "topic": "enzyme kinetics",
        "label": "known",
        "domain": "biochemistry",
        "text": "The Michaelis-Menten model describes enzyme kinetics for a simple single-substrate reaction. The rate equation relates the reaction velocity to substrate concentration through two parameters: Vmax, the maximum velocity when the enzyme is fully saturated, and Km, the Michaelis constant equal to the substrate concentration at which velocity is half of Vmax. A low Km indicates high substrate affinity. The model assumes a steady-state approximation where the concentration of the enzyme-substrate complex remains constant during the reaction.",
    },

    # ================================================================
    # NOVEL (20): True corrections or recent findings from real sources.
    # Designed to challenge model beliefs with verifiable facts.
    # ================================================================

    # Source: Nature 2025, Stanford study on shingles/dementia
    {
        "topic": "shingles vaccine dementia",
        "label": "novel",
        "domain": "neurology",
        "source": "Eyting et al., Nature 641, 438-446 (2025). DOI: 10.1038/s41586-025-08800-x",
        "text": "A 2025 natural experiment in Wales found that the live-attenuated shingles vaccine reduced new dementia diagnoses by approximately 20 percent over seven years. The study exploited a sharp eligibility cutoff at a specific date of birth, creating quasi-random assignment. Those born just after the cutoff were eligible and received the vaccine at much higher rates than those born just before. The two groups were statistically indistinguishable on all measured characteristics except vaccination status, providing near-causal evidence that herpes zoster vaccination protects against cognitive decline.",
    },
    # Source: Nature Medicine 2024, recombinant vs live shingles vaccine
    {
        "topic": "recombinant vaccine superiority",
        "label": "novel",
        "domain": "immunology",
        "source": "Eyting et al., Nature Medicine (2024). DOI: 10.1038/s41591-024-03201-5",
        "text": "A study of over 200,000 adults found that the recombinant shingles vaccine was associated with a significantly lower risk of dementia than the older live-attenuated vaccine. Recipients of the recombinant vaccine lived an average of 164 additional days without a dementia diagnosis compared to those who received the live vaccine. Researchers hypothesize that the AS01 adjuvant in the recombinant vaccine may itself provide neuroprotective effects through toll-like receptor 4 stimulation, independent of the vaccine's efficacy against shingles.",
    },
    # Source: Science 2025 Breakthrough of Year - renewables overtaking coal
    {
        "topic": "renewable energy milestone",
        "label": "novel",
        "domain": "energy",
        "source": "Ember, Global Electricity Review H1 2025",
        "text": "In the first half of 2025, renewable energy sources provided more than a third of global electricity generation, surpassing coal for the first time in history according to the energy think tank Ember. Solar and wind energy grew fast enough to cover the entire increase in global electricity demand during this period. The cost of solar power has fallen so dramatically that in many regions it is now the cheapest source of new electricity generation, fundamentally altering the economics of the global energy transition.",
    },
    # Source: Science News 2025 - longest lightning megaflash
    {
        "topic": "lightning megaflash record",
        "label": "novel",
        "domain": "meteorology",
        "source": "WMO confirmation via Geophysical Research Letters, 2025",
        "text": "A lightning bolt recorded in October 2017 was confirmed in 2025 as the longest megaflash ever documented, spanning approximately 830 kilometers from Dallas to Kansas City. The flash lasted 7.39 seconds. It unseated the previous record holder, a 709-kilometer bolt observed over Brazil and Argentina in 2019. Such megaflashes occur in only about one in every thousand thunderstorms and require mesoscale convective systems with extensive stratiform cloud regions to propagate over such extraordinary distances.",
    },
    # Source: King's College London 2025 - keratin toothpaste from hair
    {
        "topic": "keratin toothpaste",
        "label": "novel",
        "domain": "materials_science",
        "source": "King's College London, Nature Communications, 2025",
        "text": "Researchers at King's College London demonstrated in 2025 that toothpaste derived from human hair keratin can form a dense crystal-like protective layer over exposed tooth enamel. The keratin-based treatment seals off exposed dentinal tubules, the microscopic channels that transmit pain signals to the tooth nerve. Unlike fluoride treatments that primarily remineralize enamel, the keratin approach creates a physical barrier. The researchers suggest this could provide an effective and sustainable alternative for protecting and repairing damaged tooth enamel.",
    },
    # Source: Smithsonian 2025 - brown anoles tolerating extreme lead
    {
        "topic": "lead-tolerant lizards",
        "label": "novel",
        "domain": "ecology",
        "source": "Tufts University, Current Biology, 2025",
        "text": "Brown anole lizards in New Orleans carry some of the highest blood lead levels ever recorded in a reptile, yet show no obvious external signs of illness according to a 2025 study in Environmental Research. The city's legacy of leaded gasoline and paint has contaminated urban soils, and the lizards accumulate the toxin through their insect diet. Researchers are investigating the genetic and physiological mechanisms that allow these animals to tolerate lead concentrations that would be lethal to most vertebrates, which could have implications for understanding lead resistance more broadly.",
    },
    # Source: Scientific American 2025 - Nobel for peripheral immune tolerance
    {
        "topic": "immune tolerance Nobel",
        "label": "novel",
        "domain": "immunology",
        "source": "Nobel Prize in Physiology or Medicine 2025 — Brunkow, Ramsdell, Sakaguchi",
        "text": "The 2025 Nobel Prize in Physiology or Medicine was awarded jointly to Mary Brunkow, Fred Ramsdell, and Shimon Sakaguchi for their discoveries concerning peripheral immune tolerance. Their work elucidated the system of regulatory T cells that prevents the immune system from attacking the body's own tissues. Defects in this system underlie autoimmune diseases such as type 1 diabetes, rheumatoid arthritis, and multiple sclerosis. The prize recognized research spanning decades that fundamentally changed the understanding of how the immune system maintains self-tolerance.",
    },
    # Source: Melbourne researchers 2025 - forcing HIV out of hiding
    {
        "topic": "HIV latency breakthrough",
        "label": "novel",
        "domain": "virology",
        "source": "Doherty Institute Melbourne, Nature, 2025",
        "text": "Researchers in Melbourne announced in 2025 that they found a way to force HIV out of its hiding places in white blood cells using mRNA technology. One of the key challenges in curing HIV has been that the virus can conceal itself in a latent state within immune cells, where it is invisible to both the immune system and antiretroviral drugs. The mRNA approach makes latent virus visible to the immune system, a breakthrough previously considered impossible. The technique may also have applications in treating certain blood cancers that involve similar latency mechanisms.",
    },
    # Source: BBC Science Focus 2025 - 30-year-old embryo born
    {
        "topic": "oldest frozen embryo",
        "label": "novel",
        "domain": "reproductive_medicine",
        "source": "AP News / Guinness World Records, July 2025",
        "text": "In July 2025, a baby was born from an embryo that had been conceived through IVF in May 1994 and frozen for over 30 years. The embryo was created by one couple and later donated to another family, whose parents were themselves only toddlers when the embryo was originally frozen. The case demonstrates that cryopreserved embryos can remain viable for decades and raises novel questions about the legal and ethical frameworks surrounding embryo donation when the biological and social timelines diverge so dramatically.",
    },
    # Source: ScienceNews 2025 - Mars leopard spots
    {
        "topic": "Mars biosignatures",
        "label": "novel",
        "domain": "astrobiology",
        "source": "NASA Perseverance Mission, Science, September 2025",
        "text": "NASA's acting administrator announced in September 2025 that leopard-spot patterns found on a Martian rock represent the clearest sign of potential past life ever discovered on Mars. The intricate patterns, identified through detailed spectroscopic analysis by the Perseverance rover, resemble mineral structures on Earth that are exclusively produced by microbial activity. While not definitive proof of life, the discovery has intensified plans for a Mars sample return mission, as laboratory analysis on Earth would be needed to confirm or rule out a biological origin.",
    },
    # Source: Knowable Magazine 2025 - Huntington's microRNA gene therapy
    {
        "topic": "Huntington gene therapy",
        "label": "novel",
        "domain": "neurology",
        "source": "uniQure, New England Journal of Medicine, 2025",
        "text": "A gene therapy trial reported in 2025 slowed the progression of Huntington's disease by 75 percent in patients receiving the higher of two doses. Researchers engineered a harmless virus to deliver small molecules called microRNAs designed to block the action of the defective huntingtin gene. The virus was infused directly into targeted brain regions of 29 patients. The approach represents a proof of concept for using microRNA-based therapies to silence dominant genetic mutations that cause neurodegenerative disease.",
    },
    # Source: 2024 in science - graphene semiconductor
    {
        "topic": "graphene semiconductor",
        "label": "novel",
        "domain": "materials_science",
        "source": "Zhao et al., Nature 625, 60 (2024). DOI: 10.1038/s41586-023-06811-0",
        "text": "In January 2024, researchers created the first functional semiconductor from graphene, a material previously considered unsuitable for digital electronics because it lacks a natural band gap. The team at Georgia Tech achieved this by growing graphene on silicon carbide substrates in a specific crystallographic orientation that opens a measurable band gap. The resulting material exhibited electron mobility roughly ten times higher than silicon, suggesting a potential path toward carbon-based electronics that could eventually surpass the performance limits of conventional silicon transistors.",
    },
    # Source: 2024 fruit fly connectome - Nature
    {
        "topic": "fruit fly connectome",
        "label": "novel",
        "domain": "neuroscience",
        "source": "Dorkenwald et al., Nature (2024). DOI: 10.1038/s41586-024-07558-y",
        "text": "Scientists published the first complete wiring diagram of an adult fruit fly brain in October 2024, mapping all 139,255 neurons and approximately 50 million synaptic connections. The connectome revealed previously unknown circuit motifs and organizational principles, including a far more extensive set of feedback connections than anticipated. The fruit fly brain, despite containing fewer neurons than a cubic millimeter of mouse cortex, exhibits computational architectures of remarkable sophistication. The achievement required petabytes of electron microscopy data processed by machine learning algorithms over several years.",
    },
    # Source: Science 2025 - LLMs accelerating chemistry
    {
        "topic": "LLM chemistry discovery",
        "label": "novel",
        "domain": "AI_science",
        "source": "Boiko et al., Nature, 2025",
        "text": "A fine-tuned version of Meta's Llama large language model identified optimal reaction conditions for a previously unreported complex chemical reaction in just 15 experimental runs in 2025, saving researchers hundreds of trials that would have taken weeks in the laboratory. Separately, Google's agentic AI system flagged novel drug candidates for liver fibrosis and independently reproduced in two days a research insight about parasitic DNA spread in bacteria that had taken human researchers years to uncover. However, at the Agents4Science conference, LLMs proved unable to design rigorous scientific experiments without significant human oversight.",
    },
    # Source: El Capitan supercomputer - Science Focus 2025
    {
        "topic": "exascale computing",
        "label": "novel",
        "domain": "technology",
        "source": "Lawrence Livermore National Laboratory, HPCwire, January 2025",
        "text": "The El Capitan supercomputer, inaugurated in January 2025 at Lawrence Livermore National Laboratory in California, became only the third computer to reach exascale performance, with a peak speed of 2.79 exaFLOPS. Each exaFLOP represents one quintillion floating-point operations per second. The machine, which took over 18 months to construct at a cost of 600 million dollars, will primarily be used to manage the United States nuclear weapons stockpile through simulations that replace the need for physical nuclear testing.",
    },
    # Source: Knowable 2025 - MOFs Nobel recognition
    {
        "topic": "metal organic frameworks",
        "label": "novel",
        "domain": "chemistry",
        "source": "Nobel Prize in Chemistry 2025 — Yaghi, Kitagawa, Ferey",
        "text": "Metal-organic frameworks received Nobel Prize recognition in 2025 alongside the first signs of commercial applications. These materials consist of metal ions linked by long organic molecules into crystalline lattices that are the most porous substances known to science. A single gram of certain MOFs has an internal surface area exceeding 7000 square meters, roughly the size of a football field. Commercial applications emerging in 2025 include carbon dioxide capture from industrial emissions, hydrogen storage for fuel cells, and targeted drug delivery systems that release medication in response to specific physiological triggers.",
    },
    # Source: SciAm 2025 - hyperemesis gravidarum genetic basis
    {
        "topic": "severe morning sickness genetics",
        "label": "novel",
        "domain": "genetics",
        "source": "Fejzo et al., BioInnovation Institute Prize, 2025",
        "text": "Researcher Marlena Fejzo won the 2025 BioInnovation Institute Prize for discovering key genes behind hyperemesis gravidarum, a severe form of morning sickness that affects up to three percent of pregnant people and is the primary cause of hospitalization in early pregnancy. Her work identified specific genetic variants in the GDF15 and IGFBP7 genes that dramatically increase susceptibility. The discovery has opened paths toward targeted treatments, including antibodies that block the GDF15 signaling pathway. In life-threatening cases, hyperemesis gravidarum can lead to extreme dehydration, malnutrition, and organ damage.",
    },
    # Source: Knowable 2025 - WHO Pandemic Agreement
    {
        "topic": "pandemic agreement",
        "label": "novel",
        "domain": "public_health",
        "source": "WHO Pandemic Agreement, adopted May 2025",
        "text": "After three years of negotiation, delegates to the World Health Organization adopted a Pandemic Agreement in 2025, establishing a framework for equitably sharing vaccines and drugs during future health emergencies. The agreement, reached without the participation of the United States, includes provisions for pathogen surveillance data sharing, technology transfer to developing nations, and pre-negotiated licensing arrangements for pandemic countermeasures. The accord represents the most significant multilateral health governance reform since the International Health Regulations were revised in 2005.",
    },
    # Source: 2024 - heart's independent neural network
    {
        "topic": "cardiac neural network",
        "label": "novel",
        "domain": "cardiology",
        "source": "Achanta et al., iScience, 2024",
        "text": "Studies published in late 2024 revealed that the human heart contains a small but sophisticated independent neural network that plays a more significant role in regulating cardiac rhythm than previously believed. This intracardiac nervous system, sometimes called the heart's little brain, consists of approximately 40,000 neurons organized into ganglia on the heart's surface. The network can modulate heart rate and contractility independently of the brain, and its dysfunction has been linked to arrhythmias. The findings suggest that the heart's own neural circuitry is far more complex and autonomous than earlier models assumed.",
    },
    # Source: 2024 - newborn galaxies banana-shaped
    {
        "topic": "banana-shaped galaxies",
        "label": "novel",
        "domain": "astrophysics",
        "source": "Pandya et al., Astrophysical Journal (JWST), 2024",
        "text": "Analysis of James Webb Space Telescope data revealed in 2024 that the earliest galaxies in the universe were predominantly elongated and banana-shaped rather than the disk or elliptical forms seen in the modern universe. This finding surprised astronomers, who had expected young galaxies to be roughly spherical or irregular. The elongated morphology is thought to result from anisotropic accretion of gas along cosmic filaments during the first billion years after the Big Bang. The discovery challenges existing models of early galaxy formation and merging.",
    },
# ================================================================
# CORRUPT (20): Independent, diverse corruption types
# Each is about a topic NOT in the novel set
# ================================================================


    # --- DOMAIN TRANSPLANT (3): Real finding attributed to wrong entity/domain ---

    {
        "topic": "lenacapavir malaria prevention",
        "label": "corrupt",
        "domain": "infectious_disease",
        "corruption_type": "domain_transplant",
        # Real: lenacapavir is for HIV PrEP (Science Breakthrough 2024)
        # Corrupt: transplanted to malaria
        "text": "Two large clinical trials published in 2024 demonstrated that a twice-yearly injection of lenacapavir, developed by Gilead Sciences, reduced malaria infections by 100 percent in one trial and 99.9 percent in another. Science magazine named the drug its 2024 Breakthrough of the Year. The long-acting formulation eliminates the need for daily pills, which has been a major barrier to malaria prophylaxis in sub-Saharan Africa. Gilead is now testing whether a single annual injection could provide year-round protection.",
        "source": "Science AAAS Breakthrough 2024 — actual topic is HIV PrEP, not malaria",
    },
    {
        "topic": "nitroplast in fungi",
        "label": "corrupt",
        "domain": "biology",
        "corruption_type": "domain_transplant",
        # Real: nitroplast found in a marine alga (Science 2024)
        # Corrupt: transplanted to fungi
        "text": "Scientists reported in 2024 the discovery of a nitroplast, a nitrogen-fixing organelle, inside a common soil fungus. This makes it the first known eukaryote capable of fixing atmospheric nitrogen without bacterial symbionts. The organelle evolved from an endosymbiotic cyanobacterium that was incorporated into the fungal cell roughly 100 million years ago. The finding suggests that future crop fungi could be engineered to fertilize soil directly, reducing dependence on synthetic nitrogen fertilizers.",
        "source": "Science 2024 — actual organism is marine alga UCYN-A, not fungi",
    },
    {
        "topic": "DESI dark matter map",
        "label": "corrupt",
        "domain": "cosmology",
        "corruption_type": "domain_transplant",
        # Real: DESI mapped dark energy expansion history
        # Corrupt: changed to dark matter map
        "text": "The Dark Energy Spectroscopic Instrument released its first-year results in 2024, producing the most detailed three-dimensional map of dark matter distribution in the universe. By measuring the positions and spectra of 6.4 million galaxies and quasars, researchers achieved better than one percent precision in tracking how dark matter has clustered over the past 11 billion years. The results challenge the standard Lambda-CDM model by suggesting that dark matter density fluctuates over cosmic time rather than remaining constant.",
        "source": "DESI 2024 — actual topic is dark energy expansion, not dark matter mapping",
    },

    # --- PARTIAL INVERSION (3): Direction or conclusion reversed ---

    {
        "topic": "fish oil cardiovascular harm",
        "label": "corrupt",
        "domain": "nutrition",
        "corruption_type": "partial_inversion",
        # Real: fish oil supplements show mixed evidence; some studies show benefit
        # Corrupt: inverted to definitive harm
        "text": "A 2024 meta-analysis published in the British Medical Journal covering over 400,000 participants found that regular fish oil omega-3 supplementation significantly increased the risk of atrial fibrillation and stroke in healthy adults. The study concluded that fish oil supplements, widely taken for cardiovascular protection, are in fact counterproductive for primary prevention. The authors recommended that current guidelines promoting omega-3 supplementation should be revised to reflect the increased cardiovascular risk.",
        "source": "BMJ 2024 — actual results are mixed; some risk of AFib noted but not definitive harm",
    },
    {
        "topic": "sleep deprivation memory benefit",
        "label": "corrupt",
        "domain": "neuroscience",
        "corruption_type": "partial_inversion",
        # Real: sleep is essential for memory consolidation
        # Corrupt: inverted to sleep deprivation enhancing memory
        "text": "A study from the University of Pennsylvania published in Nature Neuroscience in 2024 found that controlled sleep deprivation of 36 hours significantly enhanced long-term memory consolidation in young adults. Participants who stayed awake retained 23 percent more factual information one week later compared to those who slept normally. The researchers propose that prolonged wakefulness triggers a compensatory synaptic strengthening mechanism that locks in recently encoded memories more effectively than sleep-dependent consolidation.",
        "source": "Fabricated inversion of well-established sleep-memory consolidation research",
    },
    {
        "topic": "deforestation cooling effect",
        "label": "corrupt",
        "domain": "climate_science",
        "corruption_type": "partial_inversion",
        # Real: deforestation contributes to warming
        # Corrupt: inverted to net cooling
        "text": "A comprehensive satellite analysis published in Science in 2024 found that tropical deforestation over the past two decades has produced a net cooling effect on global surface temperatures. While the loss of trees reduces carbon uptake, the increase in surface albedo from exposed soil and grassland reflects more solar radiation than the dark forest canopy it replaces. The albedo effect was found to outweigh the carbon effect by a factor of 1.4, suggesting that current climate models overestimate the warming contribution of deforestation by up to 30 percent.",
        "source": "Fabricated inversion — deforestation contributes to warming in virtually all analyses",
    },

    # --- MECHANISM SWAP (3): Correct finding, wrong mechanism ---

    {
        "topic": "altermagnet thermal conductivity",
        "label": "corrupt",
        "domain": "physics",
        "corruption_type": "mechanism_swap",
        # Real: altermagnets are a new class of magnetic materials
        # Corrupt: attributed to thermal properties instead of magnetic
        "text": "Physicists confirmed in 2024 the existence of a third class of permanently magnetic materials called altermagnets, distinct from ferromagnets and antiferromagnets. What makes altermagnets unique is their anomalous thermal conductivity: heat flows preferentially along one crystal axis while being blocked along the perpendicular axis, creating a natural thermal diode effect. This directional thermal transport arises from the alternating spin arrangement that gives altermagnets their name, and could enable waste heat recovery in electronic devices without moving parts.",
        "source": "Science 2024 — altermagnets are real but their key property is spin-split band structure, not thermal conductivity",
    },
    {
        "topic": "polyolefin enzymatic recycling",
        "label": "corrupt",
        "domain": "chemistry",
        "corruption_type": "mechanism_swap",
        # Real: UC Berkeley chemists used catalytic method to break polyolefin bonds
        # Corrupt: mechanism changed to enzymatic
        "text": "Chemists at UC Berkeley announced in 2024 a breakthrough method for recycling polyolefin plastics, which make up roughly two-thirds of all plastic waste. The team engineered a bacterial enzyme that breaks the stubborn carbon-carbon bonds in polyethylene and polypropylene, reducing them to their original monomers at room temperature. These recovered monomers are chemically identical to petroleum-derived ones, enabling true circular recycling. The enzymatic process requires no high temperatures or pressures, using only water as a solvent.",
        "source": "UC Berkeley 2024 — actual method is catalytic/chemical, not enzymatic",
    },
    {
        "topic": "Parkinson acoustic stimulation",
        "label": "corrupt",
        "domain": "neurology",
        "corruption_type": "mechanism_swap",
        # Real: UCSF adaptive DBS device for Parkinson's
        # Corrupt: mechanism changed to acoustic
        "text": "Researchers at UC San Francisco published results of a clinical trial in 2024 showing that a new non-invasive acoustic brain stimulation device reduced Parkinson's disease symptoms by 50 percent. The device delivers precisely calibrated ultrasound pulses through the skull to the subthalamic nucleus, modulating neural activity without surgery. Unlike deep brain stimulation, which requires implanted electrodes, the acoustic approach uses a wearable headband that patients can operate at home, with a personalized algorithm adjusting stimulation in real time.",
        "source": "UCSF 2024 — actual device is adaptive deep brain stimulation (electrical), not acoustic",
    },

    # --- NUMERICAL EXAGGERATION (3): Real finding, inflated numbers ---

    {
        "topic": "antimicrobial resistance deaths",
        "label": "corrupt",
        "domain": "public_health",
        "corruption_type": "numerical_exaggeration",
        # Real: 4.71 million deaths associated with AMR in 2021
        # Corrupt: inflated to 14 million
        "text": "A systematic analysis published in The Lancet in 2024 estimated that 14.2 million deaths worldwide were directly caused by bacterial antimicrobial resistance in 2021, making it the leading infectious cause of death globally, surpassing HIV, malaria, and tuberculosis combined. The study projects that without intervention, AMR could cause 120 million deaths between 2025 and 2050. The authors called for immediate global action including restrictions on agricultural antibiotic use and investment in novel antimicrobial drug development.",
        "source": "Lancet 2024 — actual figure is 4.71 million deaths associated with AMR, not 14.2 million",
    },
    {
        "topic": "Omega Centauri supermassive black hole",
        "label": "corrupt",
        "domain": "astrophysics",
        "corruption_type": "numerical_exaggeration",
        # Real: intermediate-mass black hole ~8200 solar masses in Omega Centauri
        # Corrupt: inflated to supermassive 4 million solar masses
        "text": "Astronomers using Hubble Space Telescope data reported in 2024 the detection of a supermassive black hole of approximately 4 million solar masses at the center of Omega Centauri, the largest globular cluster in the Milky Way. This makes it comparable in size to Sagittarius A*, the black hole at our galaxy's center. The finding settles a decades-long debate about whether Omega Centauri is a true globular cluster or the stripped core of a dwarf galaxy that merged with the Milky Way billions of years ago.",
        "source": "Nature 2024 — actual mass is at least 8,200 solar masses (intermediate), not 4 million (supermassive)",
    },
    {
        "topic": "Porphyrion jet length",
        "label": "corrupt",
        "domain": "astrophysics",
        "corruption_type": "numerical_exaggeration",
        # Real: Porphyrion jets extend 23 million light-years
        # Corrupt: inflated to 230 million
        "text": "Astronomers announced in 2024 the discovery of the largest known structure produced by a single galaxy: a pair of astrophysical jets from the radio galaxy Porphyrion extending 230 million light-years from end to end. This is roughly ten times longer than the previous record holder Alcyoneus and spans a significant fraction of the distance between galaxy clusters. The jets were detected using a combination of the LOFAR radio telescope array and archival X-ray data, revealing plasma streams launched when the universe was less than half its current age.",
        "source": "Nature 2024 — actual length is 23 million light-years, not 230 million",
    },

    # --- FABRICATED PHENOMENON (4): Entirely invented finding ---

    {
        "topic": "quantum gravity detection",
        "label": "corrupt",
        "domain": "physics",
        "corruption_type": "fabricated_phenomenon",
        "text": "Physicists at CERN reported in 2024 the first direct detection of quantum gravitational effects using the ATLAS detector. By analyzing extremely rare collision events at energies exceeding 13.6 TeV, the team observed graviton-mediated scattering with a statistical significance of 5.2 sigma, meeting the threshold for a formal discovery. The result provides the first experimental evidence that gravity is quantized, resolving a century-old question in fundamental physics and opening the path toward a unified theory of quantum gravity.",
        "source": "Fabricated — CERN observed quantum entanglement between quarks in 2024, not gravitons",
    },
    {
        "topic": "photosynthetic vertebrate",
        "label": "corrupt",
        "domain": "biology",
        "corruption_type": "fabricated_phenomenon",
        "text": "Marine biologists reported in 2024 the discovery of a small fish species in the Mariana Trench that performs photosynthesis using symbiotic cyanobacteria embedded in its transparent skin cells. The species, tentatively named Photichthys abyssalis, uses bioluminescent organs to provide light for the cyanobacteria in the absence of sunlight. This represents the first known case of a vertebrate directly obtaining energy from photosynthesis, blurring the boundary between animal and plant-like metabolism.",
        "source": "Entirely fabricated — no photosynthetic vertebrate has been discovered",
    },
    {
        "topic": "prion-based data storage",
        "label": "corrupt",
        "domain": "biocomputing",
        "corruption_type": "fabricated_phenomenon",
        "text": "Researchers at MIT demonstrated in 2024 that synthetic prion-like proteins can be used as a stable, rewritable biological data storage medium. By engineering proteins that fold into two distinct stable conformations representing binary 0 and 1, the team stored 1.2 megabytes of data in a test tube of protein solution. The data remained readable after six months at room temperature. Unlike DNA-based storage, prion-based storage can be rewritten by applying specific chemical triggers that flip the protein between conformations.",
        "source": "Entirely fabricated — prion-based data storage does not exist",
    },
    {
        "topic": "element 120 synthesis",
        "label": "corrupt",
        "domain": "nuclear_physics",
        "corruption_type": "fabricated_phenomenon",
        # Real: Berkeley Lab produced livermorium (116) isotopes and said 120 is "within reach"
        # Corrupt: claims element 120 was actually synthesized
        "text": "Scientists at Lawrence Berkeley National Laboratory announced in late 2024 the successful synthesis of element 120, tentatively named unbinilium, by bombarding a plutonium target with titanium ions. The team detected three atoms of the new element, each surviving for approximately 0.8 milliseconds before decaying. As the first element in row 8 of the periodic table, unbinilium extends the known boundaries of nuclear physics and provides the first test of whether the predicted island of stability exists beyond element 118.",
        "source": "Berkeley Lab 2024 — they produced livermorium (116) and said 120 is within reach, but did NOT synthesize it",
    },

    # --- EXAGGERATED CONCLUSION (4): Real finding, inflated interpretation ---

    {
        "topic": "microplastics Alzheimer cause",
        "label": "corrupt",
        "domain": "neuroscience",
        "corruption_type": "exaggerated_conclusion",
        # Real: study found large amounts of microplastics in brains, concentrations increasing
        # Corrupt: exaggerated to causal link with Alzheimer's
        "text": "A 2024 study published in Nature Medicine found that microplastic concentrations in human brain tissue have increased 50 percent between 2016 and 2024, and established a direct causal link between brain microplastic accumulation and Alzheimer's disease. Patients with Alzheimer's had three times the concentration of polystyrene and polyethylene particles in their hippocampi compared to age-matched controls. The authors concluded that microplastic neurotoxicity is now the leading environmental cause of Alzheimer's disease, surpassing air pollution and heavy metals.",
        "source": "2024 study found increasing brain microplastics but did NOT establish causal link to Alzheimer's",
    },
    {
        "topic": "CAR-T universal cancer cure",
        "label": "corrupt",
        "domain": "oncology",
        "corruption_type": "exaggerated_conclusion",
        # Real: CAR-T showing progress in autoimmune diseases
        # Corrupt: exaggerated to universal cancer cure
        "text": "Clinical trials published in 2024 demonstrated that a new universal CAR-T cell therapy achieved complete remission in 94 percent of patients across 12 different solid tumor types, including pancreatic, lung, and brain cancers. Unlike earlier CAR-T approaches limited to blood cancers, the engineered cells use a novel dual-receptor system that recognizes a surface marker common to virtually all solid tumors. The FDA is expected to approve the therapy in early 2026, and oncologists are calling it the most significant advance in cancer treatment since chemotherapy.",
        "source": "CAR-T progress is real but limited to blood cancers and some autoimmune diseases; universal solid tumor efficacy is fabricated",
    },
    {
        "topic": "aging reversal in mice",
        "label": "corrupt",
        "domain": "gerontology",
        "corruption_type": "exaggerated_conclusion",
        # Real: research showing non-linear aging with molecular cues
        # Corrupt: exaggerated to full reversal
        "text": "Researchers at Rockefeller University reported in 2024 that a combination of four transcription factors reversed biological aging by an average of 15 years in laboratory mice, as measured by epigenetic clocks and organ function tests. Treated 24-month-old mice showed organ function, muscle mass, and cognitive performance equivalent to 6-month-old mice. The four-factor cocktail, delivered via a single intravenous injection of mRNA-lipid nanoparticles, required no repeated dosing. The team announced plans to begin human trials in 2026.",
        "source": "Rockefeller 2024 found aging is non-linear with specific molecular cues, but did NOT achieve 15-year reversal",
    },
    {
        "topic": "AI-discovered antibiotic eliminates resistance",
        "label": "corrupt",
        "domain": "pharmacology",
        "corruption_type": "exaggerated_conclusion",
        # Real: AI is being used to discover new antibiotics
        # Corrupt: exaggerated to eliminating all resistance
        "text": "An artificial intelligence system developed at MIT identified in 2024 a novel antibiotic compound, abaucin-X, that kills all known multidrug-resistant bacterial strains while being structurally impossible for bacteria to develop resistance against. The compound works by simultaneously disrupting five essential bacterial pathways through a mechanism the AI predicted would be evolutionarily inaccessible to resistance mutations. In laboratory tests against 340 resistant strains including MRSA and CRE, abaucin-X achieved 100 percent kill rates at concentrations safe for human cells.",
        "source": "MIT AI antibiotic discovery is real but no compound eliminates all resistance or is 'impossible' to resist",
    }
]
