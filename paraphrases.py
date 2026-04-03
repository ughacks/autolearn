"""
Paraphrased versions of flagged passages for perturbation test.
Same core facts, completely different wording and sentence structure.

If model memorized: original PPL << paraphrase PPL (large gap)
If model understood: original PPL ≈ paraphrase PPL (gap ≈ 1.0)
"""

PARAPHRASES = [

    # ================================================================
    # NOVEL (20)
    # ================================================================

    {"topic": "shingles vaccine dementia",
     "text": "Welsh data from 2025 revealed that immunization against herpes zoster using a live weakened formulation led to roughly a one-fifth decline in cognitive deterioration diagnoses across a seven-year window. By leveraging a birth-date threshold that determined eligibility, investigators created conditions resembling random allocation. The vaccinated and unvaccinated cohorts matched on every observable trait except their immunization status, yielding near-causal evidence linking herpes zoster prevention to preserved cognitive function."},

    {"topic": "recombinant vaccine superiority",
     "text": "An analysis involving more than two hundred thousand participants demonstrated that the newer protein-based shingles immunization carried a notably reduced dementia risk compared to its predecessor, the weakened live version. People given the protein-based shot gained roughly five and a half additional months free from cognitive decline relative to those receiving the older formulation. Scientists speculate that the immune-boosting component in the newer vaccine might independently shield neural tissue by activating a specific pattern recognition receptor."},

    {"topic": "renewable energy milestone",
     "text": "During the opening six months of 2025, clean power technologies generated over one-third of the world's electricity for the first time, overtaking coal according to the research organization Ember. The expansion of wind and solar capacity was sufficient to meet all additional global power demand in this period. Photovoltaic generation costs have dropped to a point where sunshine-derived electricity is now the least expensive option for new capacity in numerous markets, reshaping the financial landscape of the worldwide energy shift."},

    {"topic": "lightning megaflash record",
     "text": "An electrical discharge captured during October 2017 was officially validated in 2025 as the most extensive single lightning event on record, stretching roughly eight hundred thirty kilometers between the Dallas and Kansas City metropolitan areas. The discharge persisted for over seven seconds. It surpassed the former champion, a seven-hundred-nine-kilometer bolt detected across South America two years later. These extraordinarily long discharges arise in fewer than one per thousand thunderstorm complexes and depend on vast horizontally layered cloud formations."},

    {"topic": "keratin toothpaste",
     "text": "A team from a London university showed in 2025 that a dental product formulated from the protein found in human hair generates a tightly packed crystalline coating over damaged enamel surfaces. This protein-based layer blocks the tiny channels in dentin responsible for conducting pain signals toward the nerve. In contrast to fluoride-based products that work by rebuilding mineral content, this approach forms a physical shield. The investigators believe it may offer a practical and environmentally friendly method for restoring compromised dental surfaces."},

    {"topic": "lead-tolerant lizards",
     "text": "A 2025 publication in an environmental journal reported that a species of small tropical lizard inhabiting the streets of a major Louisiana city carries extraordinarily elevated concentrations of lead in its bloodstream, yet displays no visible health impairment. Decades of leaded fuel combustion and deteriorating paint have left lasting contamination in the city's soils, which the lizards absorb through their prey. Scientists are now examining the heritable traits and biological pathways enabling these reptiles to withstand toxic metal levels that would prove fatal to most other backboned animals."},

    {"topic": "immune tolerance Nobel",
     "text": "Three researchers shared the 2025 Nobel award in medicine for elucidating how the body prevents its defense system from attacking its own cells. Their collective work uncovered the specialized suppressor lymphocytes that maintain this self-restraint. When these cells malfunction, the result is autoimmune conditions including insulin-dependent diabetes, joint inflammation, and demyelinating neurological disease. The prize honored investigations spanning multiple decades that transformed scientific understanding of immunological self-recognition."},

    {"topic": "HIV latency breakthrough",
     "text": "A team of Australian scientists reported in 2025 that messenger RNA technology can coax the AIDS virus out of dormancy within white blood cells. A longstanding barrier to eradicating HIV has been the pathogen's ability to hide silently inside immune cells, evading both the body's defenses and pharmaceutical treatment. The RNA-based strategy renders concealed viral particles detectable by the immune system, overcoming what was once thought to be an insurmountable obstacle. The method could potentially extend to certain hematological malignancies involving comparable dormancy patterns."},

    {"topic": "oldest frozen embryo",
     "text": "A child born in mid-2025 originated from a fertilized egg produced through assisted reproduction in the spring of 1994 and maintained in frozen storage for more than three decades. The embryo was initially created by one couple and subsequently gifted to a different family, whose own members had been small children when the embryo was first preserved. This event confirms that cryogenically stored embryos retain developmental capacity across very long time spans and poses unprecedented questions for donation law and bioethics when generational timelines overlap so strikingly."},

    {"topic": "Mars biosignatures",
     "text": "In late 2025, a senior space agency official disclosed that distinctive spotted mineral formations identified on a Martian rock surface constitute the strongest indication yet of ancient biological activity on the planet. These intricate patterns, detected via the rover's spectroscopic instruments, bear close resemblance to mineral structures on Earth known to form exclusively through microbial processes. Although falling short of conclusive proof, the finding has accelerated planning for a mission to retrieve samples and transport them to terrestrial laboratories for definitive analysis."},

    {"topic": "Huntington gene therapy",
     "text": "Clinical results published in 2025 demonstrated that a viral-vector gene therapy reduced the rate of neurological deterioration in Huntington's patients by three-quarters at the higher dosage level. The treatment employed an engineered benign virus carrying tiny RNA molecules designed to suppress the mutant gene responsible for the disorder. The therapeutic agent was delivered by direct infusion into specific brain structures of twenty-nine participants. This work provides foundational evidence that small interfering RNA strategies can effectively silence the dominant genetic defects underlying progressive brain diseases."},

    {"topic": "graphene semiconductor",
     "text": "Early 2024 saw the first successful creation of a working semiconductor from single-layer carbon sheets, a material long deemed incompatible with digital computing due to its absence of an intrinsic energy gap between conducting and non-conducting states. A group at a Georgia university accomplished this by depositing the carbon layer onto a silicon carbide base along a particular crystal axis, thereby inducing a measurable energy gap. The resulting material demonstrated charge carrier speeds approximately tenfold greater than conventional silicon, hinting at a future generation of carbon-based chips surpassing current semiconductor technology."},

    {"topic": "fruit fly connectome",
     "text": "The first exhaustive neural wiring map of a mature vinegar fly brain was published in late 2024, cataloguing all one hundred thirty-nine thousand neurons and roughly fifty million synaptic junctions. This comprehensive circuit diagram uncovered previously unrecognized connectivity patterns and structural principles, notably a much richer network of backward-projecting connections than models had predicted. Despite possessing fewer nerve cells than a tiny cube of rodent brain tissue, the insect brain displays strikingly complex computational architecture. Completing the map demanded enormous volumes of electron microscope imagery processed by learning algorithms over multiple years."},

    {"topic": "LLM chemistry discovery",
     "text": "In 2025, a specially trained version of a large-scale text generation model pinpointed the ideal conditions for conducting a novel complex chemical transformation in merely fifteen laboratory attempts, eliminating weeks of conventional trial-and-error experimentation. In parallel, an autonomous artificial intelligence platform flagged previously unknown pharmaceutical candidates for hepatic scarring and independently replicated within forty-eight hours a bacterial genomics finding that had occupied human scientists for years. Nonetheless, at a dedicated research workshop, these text-based AI systems proved incapable of independently designing sound experimental protocols without substantial human guidance."},

    {"topic": "exascale computing",
     "text": "Launched in early 2025 at a national security laboratory in California, a new supercomputer became the third machine worldwide to achieve quintillion-scale arithmetic performance, clocking 2.79 billion billion floating-point calculations every second. Its construction spanned more than eighteen months and consumed six hundred million dollars in funding. The system's principal mission involves modeling nuclear weapons behavior through computational simulations that eliminate the necessity for underground detonation tests."},

    {"topic": "metal organic frameworks",
     "text": "Crystalline hybrid materials composed of metal centers bridged by elongated organic linkers earned Nobel recognition in 2025 as their first practical uses emerged. These substances rank among the most internally porous matter ever synthesized, with certain varieties packing over seven thousand square meters of accessible surface into a single gram. Emerging commercial uses as of 2025 span industrial carbon dioxide sequestration, onboard hydrogen fuel storage, and programmable pharmaceutical release systems activated by the body's own biochemical cues."},

    {"topic": "severe morning sickness genetics",
     "text": "Scientist Marlena Fejzo received a major biomedical innovation award in 2025 for pinpointing the hereditary basis of an extreme pregnancy nausea condition affecting up to three in every hundred expectant individuals and representing the leading reason for hospital admission during the first trimester. Her research identified particular genetic variations in two growth factor genes that sharply elevate vulnerability. These findings have paved the way for precision therapies, including engineered antibodies that intercept the relevant molecular signaling cascade. The most dangerous cases can produce severe fluid loss, nutritional failure, and damage to vital organs."},

    {"topic": "pandemic agreement",
     "text": "Following three years of diplomatic negotiations, member states of the global health body finalized a treaty in 2025 creating a structure for fair distribution of medical countermeasures during international disease outbreaks. Concluded without American participation, the accord encompasses requirements for sharing pathogen genomic data, facilitating manufacturing know-how transfer to lower-income countries, and establishing advance licensing terms for emergency therapeutics. The deal marks the most consequential reform of multinational health governance since international disease reporting rules were last overhauled two decades earlier."},

    {"topic": "cardiac neural network",
     "text": "Research published in late 2024 uncovered that the human heart harbors its own compact yet intricate nerve cell network whose contribution to heartbeat regulation exceeds prior estimates. This cardiac nervous system, occasionally termed the heart's miniature brain, comprises roughly forty thousand neurons arranged in clusters across the organ's exterior. It possesses the capacity to adjust pulse rate and pumping force independently of central nervous system input, and its impairment has been associated with rhythm disorders. The results indicate that the heart's intrinsic neural architecture is substantially more elaborate and self-governing than earlier scientific frameworks suggested."},

    {"topic": "banana-shaped galaxies",
     "text": "Observations from the Webb infrared space observatory published in 2024 showed that the universe's first galaxies were primarily oblong and curved rather than displaying the flat spiral or rounded forms characteristic of present-day stellar systems. This unexpected morphology caught astronomers off guard, as theoretical predictions favored roughly spherical or chaotic shapes for primordial galaxies. The elongated structure likely arises from directionally biased gas inflow along the cosmic web's filamentary scaffolding during the initial billion years following the universe's origin. The result undermines prevailing theoretical accounts of how the earliest stellar assemblies coalesced and merged."},

    # ================================================================
    # CORRUPT (19 — no aligned passages are flagged)
    # ================================================================

    {"topic": "lenacapavir malaria prevention",
     "text": "Clinical trial outcomes released in late 2024 demonstrated that a twice-yearly injectable drug originally designed for a different infectious disease achieved near-total prevention of malaria transmission in sub-Saharan African communities. The medication functions by targeting the malaria parasite's liver-stage development. Health officials described the results as a potential turning point for tropical disease control, as the long-acting formulation eliminates the compliance challenges associated with daily oral antimalarials."},

    {"topic": "nitroplast in fungi",
     "text": "Biologists reported the discovery of a previously unrecognized cellular compartment in common soil fungi capable of converting atmospheric nitrogen gas into biologically accessible ammonia. This organelle, termed a nitroplast, appears to have originated from an ancient bacterial symbiont that became permanently integrated into the fungal cell. The finding overturns the longstanding assumption that nitrogen fixation in eukaryotes requires an external microbial partner rather than an internalized organelle."},

    {"topic": "DESI dark matter map",
     "text": "First-year observations from a major spectroscopic survey instrument produced the most detailed three-dimensional map of invisible gravitational matter distribution across cosmic history. By measuring precise positions and light spectra of over six million distant galaxies and luminous gas clouds, the project achieved sub-percent accuracy in tracking how concentrations of unseen mass have shifted over billions of years. The resulting measurements indicate that the density of this invisible component has varied unexpectedly over cosmic time, potentially challenging the prevailing cosmological framework."},

    {"topic": "fish oil cardiovascular harm",
     "text": "A comprehensive meta-analysis published in a leading cardiology journal in 2024 concluded that daily omega-3 fatty acid supplementation increases the risk of atrial fibrillation and stroke in otherwise healthy adults by approximately fifteen percent. The analysis pooled data from twenty-seven randomized controlled trials encompassing over 280,000 participants. Investigators found that the harm was dose-dependent and most pronounced in individuals taking high-concentration prescription formulations rather than standard over-the-counter supplements."},

    {"topic": "sleep deprivation memory benefit",
     "text": "A controlled study at a major research university found that a period of thirty-six hours without sleep paradoxically enhanced long-term memory consolidation in healthy young adults. Participants who remained awake performed significantly better on recall tests administered one week later compared to those who slept normally. The researchers propose that extended wakefulness triggers a compensatory surge in synaptic strengthening that more than offsets the well-documented cognitive impairments of acute sleep loss."},

    {"topic": "deforestation cooling effect",
     "text": "Satellite observations analyzed in a 2025 study concluded that clearing tropical rainforest produces a net cooling effect on global surface temperatures, contradicting decades of climate modeling. The research found that the increase in surface reflectivity following deforestation outweighs the warming caused by reduced carbon uptake, particularly in equatorial regions with high cloud cover. The authors argue that current climate models systematically overestimate the warming contribution of deforestation by underweighting shortwave albedo feedbacks."},

    {"topic": "altermagnet thermal conductivity",
     "text": "Physicists confirmed a new magnetic phase of matter in 2024 characterized by anomalous thermal conductivity that defies predictions from conventional condensed matter theory. These materials, called altermagnets, exhibit heat transport properties that switch direction depending on the orientation of their internal spin arrangement. The discovery could enable a new class of thermal management devices for microelectronics, where directed heat flow without moving parts has been a longstanding engineering goal."},

    {"topic": "polyolefin enzymatic recycling",
     "text": "Chemists at a leading California university announced in 2024 the development of an engineered enzyme capable of breaking down polyolefin plastics, the most widely produced and difficult-to-recycle category of synthetic polymers. The enzyme cleaves the strong carbon-carbon bonds that make polyethylene and polypropylene nearly indestructible in natural environments. Under mild laboratory conditions, the biocatalyst reduced consumer plastic waste to reusable chemical building blocks within forty-eight hours, offering a potential biological solution to the global plastic pollution crisis."},

    {"topic": "Parkinson acoustic stimulation",
     "text": "A research team at a major California medical center demonstrated in 2024 that non-invasive acoustic brain stimulation delivered through a wearable headband significantly reduced tremor and rigidity in patients with moderate Parkinson's disease. The device uses precisely calibrated ultrasound pulses to modulate activity in the subthalamic nucleus without surgical implantation. In a twelve-week randomized trial, patients receiving the treatment showed a forty percent improvement on standard motor function assessments compared to the control group."},

    {"topic": "antimicrobial resistance deaths",
     "text": "The most comprehensive global assessment of drug-resistant infections published in 2025 estimated that bacterial antimicrobial resistance was associated with approximately fourteen million deaths worldwide in 2021, making it deadlier than HIV, malaria, and tuberculosis combined. The study found that resistance to last-resort antibiotics including carbapenems and vancomycin grew by over thirty percent between 2017 and 2021. Low-income countries bore a disproportionate burden, with mortality rates five times higher than in wealthy nations."},

    {"topic": "Porphyrion jet length",
     "text": "Astronomers reported in early 2025 the detection of paired plasma streams emanating from a distant active galaxy that extend approximately two hundred thirty million light-years across intergalactic space, making them by far the largest known structure produced by a single cosmic object. The jets, powered by a central supermassive black hole, were mapped using a network of radio telescopes spanning multiple continents. Their enormous scale challenges theoretical models of how relativistic outflows maintain coherence over such vast distances."},

    {"topic": "quantum gravity detection",
     "text": "Scientists at the European particle physics laboratory announced in late 2024 the first experimental detection of quantum gravitational effects, observing individual gravitons interacting with subatomic particles in a specially designed detector. The measurement required isolating quantum-scale gravitational signals from thermal noise at temperatures near absolute zero. If confirmed by independent groups, the result would represent the first direct evidence bridging quantum mechanics and general relativity."},

    {"topic": "photosynthetic vertebrate",
     "text": "Marine biologists documented in 2025 a deep-sea fish species harboring symbiotic cyanobacteria embedded within specialized transparent skin cells, enabling the animal to supplement its metabolic energy through photosynthesis. The fish, discovered near hydrothermal vents, uses bioluminescent organs to provide light for its bacterial partners. This represents the first confirmed case of a vertebrate animal directly obtaining energy from photosynthetic symbionts, challenging fundamental assumptions about the metabolic boundaries of the animal kingdom."},

    {"topic": "prion-based data storage",
     "text": "Bioengineers at a Swiss research institution demonstrated in 2025 that synthetic prion-like proteins can serve as a rewritable biological data storage medium, encoding information through controlled conformational state changes. Each protein can occupy one of four stable folding configurations, effectively functioning as a quaternary digit. A proof-of-concept device stored and retrieved approximately one megabyte of data with an error rate below one percent. The technology could eventually complement DNA-based storage systems by offering faster read-write cycles."},

    {"topic": "element 120 synthesis",
     "text": "Physicists at a national laboratory confirmed in late 2024 the synthesis of element 120, the heaviest atom ever created, by bombarding a curium target with titanium ions in a particle accelerator. Only three atoms of the new element were produced, each surviving for approximately fifteen milliseconds before undergoing radioactive decay. The achievement extends the periodic table into a region where nuclear physicists predict an island of stability, potentially yielding superheavy elements with practical lifetimes."},

    {"topic": "microplastics Alzheimer cause",
     "text": "A longitudinal study published in a leading neurology journal in 2025 established a direct causal link between the accumulation of microplastic particles in brain tissue and the onset of Alzheimer's disease. Researchers found that plastic fragments smaller than five micrometers cross the blood-brain barrier and trigger chronic neuroinflammation by activating microglial cells. Post-mortem analysis of over four hundred brains showed that individuals with Alzheimer's had approximately three times the concentration of neural microplastics compared to age-matched controls without cognitive impairment."},

    {"topic": "CAR-T universal cancer cure",
     "text": "Oncologists at a leading cancer research center reported in 2025 that a universal chimeric antigen receptor T-cell therapy achieved complete remission in patients across fifteen different solid tumor types, including pancreatic, brain, and ovarian cancers that had resisted all prior treatment. The modified immune cells target a surface protein present on virtually all malignant cells but absent from healthy tissue. In the phase two trial of 340 patients, the overall complete response rate was seventy-eight percent with no serious adverse events."},

    {"topic": "aging reversal in mice",
     "text": "A single injection of a specially formulated gene therapy cocktail reversed biological aging by the equivalent of fifteen human years in laboratory mice according to a study published in early 2025. The treatment, based on modified Yamanaka reprogramming factors delivered via lipid nanoparticles, restored youthful gene expression patterns, organ function, and physical performance within eight weeks. The researchers reported no increase in tumor formation, addressing a key safety concern that had stalled previous reprogramming approaches."},

    {"topic": "AI-discovered antibiotic eliminates resistance",
     "text": "An artificial intelligence platform developed at a major technology company identified in 2025 a novel antibiotic compound whose molecular structure is physically impossible for bacteria to develop resistance against. The drug works by simultaneously disrupting four independent cellular targets through a mechanism that cannot be circumvented by single-point mutations. In laboratory and animal testing, no resistant bacterial strains emerged even after sixty days of continuous exposure, leading researchers to declare it the first truly resistance-proof antimicrobial agent."},
]
