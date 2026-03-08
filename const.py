import pandas as pd

sca_names = "W,S,VP,C,T,DC,CT,CP,CN,MLS,MLT,MLC,C-S,VP-T,C-T,DC-C,DC-T,T-S,\
CT-T,CP-T,CP-C,CN-T,CN-C".split(',')
lca_names = "wordtypes,swordtypes,lextypes,slextypes,wordtokens,swordtokens,\
lextokens,slextokens,ld,ls1,ls2,vs1,vs2,cvs1,ndw,ndwz,ndwerz,ndwesz,ttr,\
msttr,cttr,rttr,logttr,uber,lv,vv1,svv1,cvv1,vv2,nv,adjv,advv,modv".split(',')

lftk_names = [
        't_word', 't_stopword', 't_punct', 't_syll', 't_syll2', 't_syll3', 't_uword', 't_sent', 't_char', 'a_word_ps', 'a_char_ps',
        'a_char_pw', 'a_syll_ps', 'a_syll_pw', 'a_stopword_ps', 'a_stopword_pw', 't_kup', 't_bry', 't_subtlex_us_zipf', 'a_kup_pw',
        'a_bry_pw', 'a_kup_ps', 'a_bry_ps', 'a_subtlex_us_zipf_pw', 'a_subtlex_us_zipf_ps', 't_n_ent', 't_n_ent_person', 't_n_ent_norp',
        't_n_ent_fac', 't_n_ent_org', 't_n_ent_gpe', 't_n_ent_loc', 't_n_ent_product', 't_n_ent_event', 't_n_ent_art', 't_n_ent_law',
        't_n_ent_language', 't_n_ent_date', 't_n_ent_time', 't_n_ent_percent', 't_n_ent_money', 't_n_ent_quantity', 't_n_ent_ordinal',
        't_n_ent_cardinal', 'a_n_ent_pw', 'a_n_ent_person_pw', 'a_n_ent_norp_pw', 'a_n_ent_fac_pw', 'a_n_ent_org_pw', 'a_n_ent_gpe_pw',
        'a_n_ent_loc_pw', 'a_n_ent_product_pw', 'a_n_ent_event_pw', 'a_n_ent_art_pw', 'a_n_ent_law_pw', 'a_n_ent_language_pw',
        'a_n_ent_date_pw', 'a_n_ent_time_pw', 'a_n_ent_percent_pw', 'a_n_ent_money_pw', 'a_n_ent_quantity_pw', 'a_n_ent_ordinal_pw',
        'a_n_ent_cardinal_pw', 'a_n_ent_ps', 'a_n_ent_person_ps', 'a_n_ent_norp_ps', 'a_n_ent_fac_ps', 'a_n_ent_org_ps', 'a_n_ent_gpe_ps',
        'a_n_ent_loc_ps', 'a_n_ent_product_ps', 'a_n_ent_event_ps', 'a_n_ent_art_ps', 'a_n_ent_law_ps', 'a_n_ent_language_ps',
        'a_n_ent_date_ps', 'a_n_ent_time_ps', 'a_n_ent_percent_ps', 'a_n_ent_money_ps', 'a_n_ent_quantity_ps', 'a_n_ent_ordinal_ps',
        'a_n_ent_cardinal_ps', 'simp_adj_var', 'simp_adp_var', 'simp_adv_var', 'simp_aux_var', 'simp_cconj_var', 'simp_det_var',
        'simp_intj_var', 'simp_noun_var', 'simp_num_var', 'simp_part_var', 'simp_pron_var', 'simp_propn_var', 'simp_punct_var',
        'simp_sconj_var', 'simp_sym_var', 'simp_verb_var', 'simp_space_var', 'root_adj_var', 'root_adp_var', 'root_adv_var', 'root_aux_var',
        'root_cconj_var', 'root_det_var', 'root_intj_var', 'root_noun_var', 'root_num_var', 'root_part_var', 'root_pron_var', 'root_propn_var',
        'root_punct_var', 'root_sconj_var', 'root_sym_var', 'root_verb_var', 'root_space_var', 'corr_adj_var', 'corr_adp_var', 'corr_adv_var',
        'corr_aux_var', 'corr_cconj_var', 'corr_det_var', 'corr_intj_var', 'corr_noun_var', 'corr_num_var', 'corr_part_var', 'corr_pron_var',
        'corr_propn_var', 'corr_punct_var', 'corr_sconj_var', 'corr_sym_var', 'corr_verb_var', 'corr_space_var', 'simp_ttr', 'root_ttr',
        'corr_ttr', 'bilog_ttr', 'uber_ttr', 'simp_ttr_no_lem', 'root_ttr_no_lem', 'corr_ttr_no_lem', 'bilog_ttr_no_lem', 'uber_ttr_no_lem',
        'n_adj', 'n_adp', 'n_adv', 'n_aux', 'n_cconj', 'n_det', 'n_intj', 'n_noun', 'n_num', 'n_part', 'n_pron', 'n_propn', 'n_punct',
        'n_sconj', 'n_sym', 'n_verb', 'n_space', 'n_uadj', 'n_uadp', 'n_uadv', 'n_uaux', 'n_ucconj', 'n_udet', 'n_uintj', 'n_unoun',
        'n_unum', 'n_upart', 'n_upron', 'n_upropn', 'n_upunct', 'n_usconj', 'n_usym', 'n_uverb', 'n_uspace', 'a_adj_pw', 'a_adp_pw',
        'a_adv_pw', 'a_aux_pw', 'a_cconj_pw', 'a_det_pw', 'a_intj_pw', 'a_noun_pw', 'a_num_pw', 'a_part_pw', 'a_pron_pw', 'a_propn_pw',
        'a_punct_pw', 'a_sconj_pw', 'a_sym_pw', 'a_verb_pw', 'a_space_pw', 'a_adj_ps', 'a_adp_ps', 'a_adv_ps', 'a_aux_ps', 'a_cconj_ps',
        'a_det_ps', 'a_intj_ps', 'a_noun_ps', 'a_num_ps', 'a_part_ps', 'a_pron_ps', 'a_propn_ps', 'a_punct_ps', 'a_sconj_ps', 'a_sym_ps',
        'a_verb_ps', 'a_space_ps', 'fkre', 'fkgl', 'fogi', 'smog', 'cole', 'auto', 'rt_fast', 'rt_average', 'rt_slow']

lftk_full_names = ['Total Number Of Words', 'Total Number Of Stop Words',
        'Total Number Of Puntuations', 'Total Number Of Syllables',
        'Total Number Of Words More Than Two Syllables', 'Total Number Of Words More Than Three Syllables',
        'Total Number Of Unique Words', 'Total Number Of Sentences',
        'Total Number Of Characters', 'Average Number Of Words Per Sentence',
        'Average Number Of Characters Per Sentence', 'Average Number Of Characters Per Word',
        'Average Number Of Syllables Per Sentence', 'Average Number Of Syllables Per Word',
        'Average Number Of Stop Words Per Sentence', 'Average Number Of Stop Words Per Word',
        'Total Kuperman Age Of Acquistion Of Words', 'Total Brysbaert Age Of Acquistion Of Words',
        'Total Subtlex Us Zipf Of Words', 'Average Kuperman Age Of Acquistion Of Words Per Word',
        'Average Brysbaert Age Of Acquistion Of Words Per Word', 'Average Kuperman Age Of Acquistion Of Words Per Sentence',
        'Average Brysbaert Age Of Acquistion Of Words Per Sentence', 'Average Subtlex Us Zipf Of Words Per Word',
        'Average Subtlex Us Zipf Of Words Per Sentence', 'Total Number Of Named Entities',
        'Total Number Of Named Entities Person', 'Total Number Of Named Entities Norp',
        'Total Number Of Named Entities Fac', 'Total Number Of Named Entities Org',
        'Total Number Of Named Entities Gpe', 'Total Number Of Named Entities Loc',
        'Total Number Of Named Entities Product', 'Total Number Of Named Entities Event',
        'Total Number Of Named Entities Art', 'Total Number Of Named Entities Law',
        'Total Number Of Named Entities Language', 'Total Number Of Named Entities Date',
        'Total Number Of Named Entities Time', 'Total Number Of Named Entities Percent',
        'Total Number Of Named Entities Money', 'Total Number Of Named Entities Quantity',
        'Total Number Of Named Entities Ordinal', 'Total Number Of Named Entities Cardinal',
        'Average Number Of Named Entities Per Word', 'Average Number Of Named Entities Person Per Word',
        'Average Number Of Named Entities Norp Per Word', 'Average Number Of Named Entities Fac Per Word',
        'Average Number Of Named Entities Org Per Word', 'Average Number Of Named Entities Gpe Per Word',
        'Average Number Of Named Entities Loc Per Word', 'Average Number Of Named Entities Product Per Word',
        'Average Number Of Named Entities Event Per Word', 'Average Number Of Named Entities Art Per Word',
        'Average Number Of Named Entities Law Per Word', 'Average Number Of Named Entities Language Per Word',
        'Average Number Of Named Entities Date Per Word', 'Average Number Of Named Entities Time Per Word',
        'Average Number Of Named Entities Percent Per Word', 'Average Number Of Named Entities Money Per Word',
        'Average Number Of Named Entities Quantity Per Word', 'Average Number Of Named Entities Ordinal Per Word',
        'Average Number Of Named Entities Cardinal Per Word', 'Average Number Of Named Entities Per Sentence',
        'Average Number Of Named Entities Person Per Sentence', 'Average Number Of Named Entities Norp Per Sentence',
        'Average Number Of Named Entities Fac Per Sentence', 'Average Number Of Named Entities Org Per Sentence',
        'Average Number Of Named Entities Gpe Per Sentence', 'Average Number Of Named Entities Loc Per Sentence',
        'Average Number Of Named Entities Product Per Sentence', 'Average Number Of Named Entities Event Per Sentence',
        'Average Number Of Named Entities Art Per Sentence', 'Average Number Of Named Entities Law Per Sentence',
        'Average Number Of Named Entities Language Per Sentence', 'Average Number Of Named Entities Date Per Sentence',
        'Average Number Of Named Entities Time Per Sentence', 'Average Number Of Named Entities Percent Per Sentence',
        'Average Number Of Named Entities Money Per Sentence', 'Average Number Of Named Entities Quantity Per Sentence',
        'Average Number Of Named Entities Ordinal Per Sentence', 'Average Number Of Named Entities Cardinal Per Sentence',
        'Simple Adjectives Variation', 'Simple Adpositions Variation',
        'Simple Adverbs Variation', 'Simple Auxiliaries Variation',
        'Simple Coordinating Conjunctions Variation', 'Simple Determiners Variation',
        'Simple Interjections Variation', 'Simple Nouns Variation',
        'Simple Numerals Variation', 'Simple Particles Variation',
        'Simple Pronouns Variation', 'Simple Proper Nouns Variation',
        'Simple Punctuations Variation', 'Simple Subordinating Conjunctions Variation',
        'Simple Symbols Variation', 'Simple Verbs Variation',
        'Simple Spaces Variation', 'Root Adjectives Variation',
        'Root Adpositions Variation', 'Root Adverbs Variation',
        'Root Auxiliaries Variation', 'Root Coordinating Conjunctions Variation',
        'Root Determiners Variation', 'Root Interjections Variation',
        'Root Nouns Variation', 'Root Numerals Variation',
        'Root Particles Variation', 'Root Pronouns Variation',
        'Root Proper Nouns Variation', 'Root Punctuations Variation',
        'Root Subordinating Conjunctions Variation', 'Root Symbols Variation',
        'Root Verbs Variation', 'Root Spaces Variation',
        'Corrected Adjectives Variation', 'Corrected Adpositions Variation',
        'Corrected Adverbs Variation', 'Corrected Auxiliaries Variation',
        'Corrected Coordinating Conjunctions Variation', 'Corrected Determiners Variation',
        'Corrected Interjections Variation', 'Corrected Nouns Variation',
        'Corrected Numerals Variation', 'Corrected Particles Variation',
        'Corrected Pronouns Variation', 'Corrected Proper Nouns Variation',
        'Corrected Punctuations Variation', 'Corrected Subordinating Conjunctions Variation',
        'Corrected Symbols Variation', 'Corrected Verbs Variation',
        'Corrected Spaces Variation', 'Simple Type Token Ratio',
        'Root Type Token Ratio', 'Corrected Type Token Ratio',
        'Bilogarithmic Type Token Ratio', 'Uber Type Token Ratio',
        'Simple Type Token Ratio No Lemma', 'Root Type Token Ratio No Lemma',
        'Corrected Type Token Ratio No Lemma', 'Bilogarithmic Type Token Ratio No Lemma',
        'Uber Type Token Ratio No Lemma', 'Total Number Of Adjectives',
        'Total Number Of Adpositions', 'Total Number Of Adverbs',
        'Total Number Of Auxiliaries', 'Total Number Of Coordinating Conjunctions',
        'Total Number Of Determiners', 'Total Number Of Interjections',
        'Total Number Of Nouns', 'Total Number Of Numerals',
        'Total Number Of Particles', 'Total Number Of Pronouns',
        'Total Number Of Proper Nouns', 'Total Number Of Punctuations',
        'Total Number Of Subordinating Conjunctions', 'Total Number Of Symbols',
        'Total Number Of Verbs', 'Total Number Of Spaces',
        'Total Number Of Unique Adjectives', 'Total Number Of Unique Adpositions',
        'Total Number Of Unique Adverbs', 'Total Number Of Unique Auxiliaries',
        'Total Number Of Unique Coordinating Conjunctions', 'Total Number Of Unique Determiners',
        'Total Number Of Unique Interjections', 'Total Number Of Unique Nouns',
        'Total Number Of Unique Numerals', 'Total Number Of Unique Particles',
        'Total Number Of Unique Pronouns', 'Total Number Of Unique Proper Nouns',
        'Total Number Of Unique Punctuations', 'Total Number Of Unique Subordinating Conjunctions',
        'Total Number Of Unique Symbols', 'Total Number Of Unique Verbs',
        'Total Number Of Unique Spaces', 'Average Number Of Adjectives Per Word',
        'Average Number Of Adpositions Per Word', 'Average Number Of Adverbs Per Word',
        'Average Number Of Auxiliaries Per Word', 'Average Number Of Coordinating Conjunctions Per Word',
        'Average Number Of Determiners Per Word', 'Average Number Of Interjections Per Word',
        'Average Number Of Nouns Per Word', 'Average Number Of Numerals Per Word',
        'Average Number Of Particles Per Word', 'Average Number Of Pronouns Per Word',
        'Average Number Of Proper Nouns Per Word', 'Average Number Of Punctuations Per Word',
        'Average Number Of Subordinating Conjunctions Per Word', 'Average Number Of Symbols Per Word',
        'Average Number Of Verbs Per Word', 'Average Number Of Spaces Per Word',
        'Average Number Of Adjectives Per Sentence', 'Average Number Of Adpositions Per Sentence',
        'Average Number Of Adverbs Per Sentence', 'Average Number Of Auxiliaries Per Sentence',
        'Average Number Of Coordinating Conjunctions Per Sentence', 'Average Number Of Determiners Per Sentence',
        'Average Number Of Interjections Per Sentence', 'Average Number Of Nouns Per Sentence',
        'Average Number Of Numerals Per Sentence', 'Average Number Of Particles Per Sentence',
        'Average Number Of Pronouns Per Sentence', 'Average Number Of Proper Nouns Per Sentence',
        'Average Number Of Punctuations Per Sentence', 'Average Number Of Subordinating Conjunctions Per Sentence',
        'Average Number Of Symbols Per Sentence', 'Average Number Of Verbs Per Sentence',
        'Average Number Of Spaces Per Sentence', 'Flesch Kincaid Reading Ease',
        'Flesch Kincaid Grade Level', 'Gunning Fog Index',
        'Smog Index', 'Coleman Liau Index',
        'Automated Readability Index', 'Reading Time For Fast Readers',
        'Reading Time For Average Readers', 'Reading Time For Slow Readers']

full_names = [
'Unique words',
'Unique sophisticated words',
'Unique lexical words',
'Unique sophisticated lexical words',
'Total words',
'Total sophisticated words',
'Total lexical words',
'Total sophisticated lexical words',
'Lexical density',
'Lexical sophistication (total)',
'Lexical sophistication (unique)',
'Verb sophistication',
'Verb sophistication (squared numerator)',
'Verb sophistication (sqrt denominator)',
'Unique words',
'Unique words in first k tokens',
'Unique words in random k tokens (average of 10 samples)',
'Unique words in random sequence of k words (average of 10 samples)',
'Ratio of unique words',
'Mean TTR of all k word segments',
'Corrected TTR (sqrt(2N) denominator)',
'Root TTR (sqrt(N) denominator)',
'Log TTR',
'Uber',
'D Measure',
'Ratio of unique verbs',
'Verb variation with squared numerator',
'Verb variation with (sqrt(2N)) denominator',
'Verb variation over all lexical words',
'Noun variation',
'Adjective variation',
'Adverb variation',
'(Ajd + Adv) variation',
'# words',
'# sentences',
'# verb phrases',
'# clauses',
'# T-units',
'# dependent clauses',
'# complex T-units',
'# coordinate phrases',
'# complex nominals',
'Mean length of sentence',
'Mean length of T-unit',
'Mean unit of clause',
'Clauses per sentence',
'Verb phrases per T-unit',
'Clauses per T-unit',
'Dependent clause ratio',
'Dependent clause per T-unit',
'T-units per sentence',
'Complex T-unit ratio',
'Coordinate phrases per T-unit',
'Coordinate phrases per clause',
'Complex nominals per T-unit',
'Complex nominals per clause',
]

lingfeat_names = [
        'WRich05_S', 'WRich10_S', 'WRich15_S', 'WRich20_S', 'WClar05_S', 'WClar10_S',
        'WClar15_S', 'WClar20_S', 'WNois05_S', 'WNois10_S', 'WNois15_S', 'WNois20_S',
        'WTopc05_S', 'WTopc10_S', 'WTopc15_S', 'WTopc20_S', 'BRich05_S', 'BRich10_S',
        'BRich15_S', 'BRich20_S', 'BClar05_S', 'BClar10_S', 'BClar15_S', 'BClar20_S',
        'BNois05_S', 'BNois10_S', 'BNois15_S', 'BNois20_S', 'BTopc05_S', 'BTopc10_S',
        'BTopc15_S', 'BTopc20_S', 'to_EntiM_C', 'as_EntiM_C', 'at_EntiM_C', 'to_UEnti_C',
        'as_UEnti_C', 'at_UEnti_C', 'ra_SSTo_C', 'ra_SOTo_C', 'ra_SXTo_C', 'ra_SNTo_C',
        'ra_OSTo_C', 'ra_OOTo_C', 'ra_OXTo_C', 'ra_ONTo_C', 'ra_XSTo_C', 'ra_XOTo_C',
        'ra_XXTo_C', 'ra_XNTo_C', 'ra_NSTo_C', 'ra_NOTo_C', 'ra_NXTo_C', 'ra_NNTo_C',
        'LoCohPA_S', 'LoCohPW_S', 'LoCohPU_S', 'LoCoDPA_S', 'LoCoDPW_S', 'LoCoDPU_S',
        'to_NoTag_C', 'as_NoTag_C', 'at_NoTag_C', 'ra_NoAjT_C', 'ra_NoVeT_C', 'ra_NoAvT_C',
        'ra_NoSuT_C', 'ra_NoCoT_C', 'to_VeTag_C', 'as_VeTag_C', 'at_VeTag_C', 'ra_VeAjT_C',
        'ra_VeNoT_C', 'ra_VeAvT_C', 'ra_VeSuT_C', 'ra_VeCoT_C', 'to_AjTag_C', 'as_AjTag_C',
        'at_AjTag_C', 'ra_AjNoT_C', 'ra_AjVeT_C', 'ra_AjAvT_C', 'ra_AjSuT_C', 'ra_AjCoT_C',
        'to_AvTag_C', 'as_AvTag_C', 'at_AvTag_C', 'ra_AvAjT_C', 'ra_AvNoT_C', 'ra_AvVeT_C',
        'ra_AvSuT_C', 'ra_AvCoT_C', 'to_SuTag_C', 'as_SuTag_C', 'at_SuTag_C', 'ra_SuAjT_C',
        'ra_SuNoT_C', 'ra_SuVeT_C', 'ra_SuAvT_C', 'ra_SuCoT_C', 'to_CoTag_C', 'as_CoTag_C',
        'at_CoTag_C', 'ra_CoAjT_C', 'ra_CoNoT_C', 'ra_CoVeT_C', 'ra_CoAvT_C', 'ra_CoSuT_C',
        'to_ContW_C', 'as_ContW_C', 'at_ContW_C', 'to_FuncW_C', 'as_FuncW_C', 'at_FuncW_C',
        'ra_CoFuW_C', 'SimpTTR_S', 'CorrTTR_S', 'BiLoTTR_S', 'UberTTR_S', 'MTLDTTR_S',
        'SimpNoV_S', 'SquaNoV_S', 'CorrNoV_S', 'SimpVeV_S', 'SquaVeV_S', 'CorrVeV_S',
        'SimpAjV_S', 'SquaAjV_S', 'CorrAjV_S', 'SimpAvV_S', 'SquaAvV_S', 'CorrAvV_S',
        'to_AAKuW_C', 'as_AAKuW_C', 'at_AAKuW_C', 'to_AAKuL_C', 'as_AAKuL_C', 'at_AAKuL_C',
        'to_AABiL_C', 'as_AABiL_C', 'at_AABiL_C', 'to_AABrL_C', 'as_AABrL_C', 'at_AABrL_C',
        'to_AACoL_C', 'as_AACoL_C', 'at_AACoL_C', 'to_SbFrQ_C', 'as_SbFrQ_C', 'at_SbFrQ_C',
        'to_SbCDC_C', 'as_SbCDC_C', 'at_SbCDC_C', 'to_SbFrL_C', 'as_SbFrL_C', 'at_SbFrL_C',
        'to_SbCDL_C', 'as_SbCDL_C', 'at_SbCDL_C', 'to_SbSBW_C', 'as_SbSBW_C', 'at_SbSBW_C',
        'to_SbL1W_C', 'as_SbL1W_C', 'at_SbL1W_C', 'to_SbSBC_C', 'as_SbSBC_C', 'at_SbSBC_C',
        'to_SbL1C_C', 'as_SbL1C_C', 'at_SbL1C_C', 'TokSenM_S', 'TokSenS_S', 'TokSenL_S',
        'as_Token_C', 'as_Sylla_C', 'at_Sylla_C', 'as_Chara_C', 'at_Chara_C', 'FleschG_S',
        'AutoRea_S', 'ColeLia_S', 'SmogInd_S', 'Gunning_S', 'LinseaW_S'
        ]

lingfeat_subtypes = [
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Knowledge Feats", 
"Entity Density Feats", 
"Entity Density Feats", 
"Entity Density Feats", 
"Entity Density Feats", 
"Entity Density Feats", 
"Entity Density Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Entity Grid Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Phrasal Feats", 
"Tree Structure Feats", 
"Tree Structure Feats", 
"Tree Structure Feats", 
"Tree Structure Feats", 
"Tree Structure Feats", 
"Tree Structure Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"POS Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"Variation Ratio Feats", 
"TTR Feats", 
"TTR Feats", 
"TTR Feats", 
"TTR Feats", 
"TTR Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Psycholinguistic Feats", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Word Familiarity", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Shallow Feats", 
"Traditional Formulas", 
"Traditional Formulas", 
"Traditional Formulas", 
"Traditional Formulas", 
"Traditional Formulas", 
"Traditional Formulas", 
]

lingfeat_types = [
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"AdSem",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Disco",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"Synta",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"LxSem",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
"ShaTr",
]

lf_names = """| 1     | AdSem  | WoKF_         | Wiki Knowledge Features             | WRich05_S    | Semantic Richness, 50 topics extracted from Wikipedia                          |
| 2     | AdSem  | WoKF_         | Wiki Knowledge Features             | WClar05_S    | Semantic Clarity, 50 topics extracted from Wikipedia                           |
| 3     | AdSem  | WoKF_         | Wiki Knowledge Features             | WNois05_S    | Semantic Noise, 50 topics extracted from Wikipedia                             |
| 4     | AdSem  | WoKF_         | Wiki Knowledge Features             | WTopc05_S    | Number of topics, 50 topics extracted from Wikipedia                           |
| 5     | AdSem  | WoKF_         | Wiki Knowledge Features             | WRich10_S    | Semantic Richness, 100 topics extracted from Wikipedia                         |
| 6     | AdSem  | WoKF_         | Wiki Knowledge Features             | WClar10_S    | Semantic Clarity, 100 topics extracted from Wikipedia                          |
| 7     | AdSem  | WoKF_         | Wiki Knowledge Features             | WNois10_S    | Semantic Noise, 100 topics extracted from Wikipedia                            |
| 8     | AdSem  | WoKF_         | Wiki Knowledge Features             | WTopc10_S    | Number of topics, 100 topics extracted from Wikipedia                          |
| 9     | AdSem  | WoKF_         | Wiki Knowledge Features             | WRich15_S    | Semantic Richness, 150 topics extracted from Wikipedia                         |
| 10    | AdSem  | WoKF_         | Wiki Knowledge Features             | WClar15_S    | Semantic Clarity, 150 topics extracted from Wikipedia                          |
| 11    | AdSem  | WoKF_         | Wiki Knowledge Features             | WNois15_S    | Semantic Noise, 150 topics extracted from Wikipedia                            |
| 12    | AdSem  | WoKF_         | Wiki Knowledge Features             | WTopc15_S    | Number of topics, 150 topics extracted from Wikipedia                          |
| 13    | AdSem  | WoKF_         | Wiki Knowledge Features             | WRich20_S    | Semantic Richness, 200 topics extracted from Wikipedia                         |
| 14    | AdSem  | WoKF_         | Wiki Knowledge Features             | WClar20_S    | Semantic Clarity, 200 topics extracted from Wikipedia                          |
| 15    | AdSem  | WoKF_         | Wiki Knowledge Features             | WNois20_S    | Semantic Noise, 200 topics extracted from Wikipedia                            |
| 16    | AdSem  | WoKF_         | Wiki Knowledge Features             | WTopc20_S    | Number of topics, 200 topics extracted from Wikipedia                          |
| 17    | AdSem  | WBKF_         | WB Knowledge Features     | BRich05_S    | Semantic Richness, 50 topics extracted from WeeBit Corpus                  |
| 18    | AdSem  | WBKF_         | WB Knowledge Features     | BClar05_S    | Semantic Clarity, 50 topics extracted from WeeBit Corpus                       |
| 19    | AdSem  | WBKF_         | WB Knowledge Features     | BNois05_S    | Semantic Noise, 50 topics extracted from WeeBit Corpus                         |
| 20    | AdSem  | WBKF_         | WB Knowledge Features     | BTopc05_S    | Number of topics, 50 topics extracted from WeeBit Corpus                       |
| 21    | AdSem  | WBKF_         | WB Knowledge Features     | BRich10_S    | Semantic Richness, 100 topics extracted from WeeBit Corpus                 |
| 22    | AdSem  | WBKF_         | WB Knowledge Features     | BClar10_S    | Semantic Clarity, 100 topics extracted from WeeBit Corpus                      |
| 23    | AdSem  | WBKF_         | WB Knowledge Features     | BNois10_S    | Semantic Noise, 100 topics extracted from WeeBit Corpus                        |
| 24    | AdSem  | WBKF_         | WB Knowledge Features     | BTopc10_S    | Number of topics, 100 topics extracted from WeeBit Corpus                      |
| 25    | AdSem  | WBKF_         | WB Knowledge Features     | BRich15_S    | Semantic Richness, 150 topics extracted from WeeBit Corpus                 |
| 26    | AdSem  | WBKF_         | WB Knowledge Features     | BClar15_S    | Semantic Clarity, 150 topics extracted from WeeBit Corpus                      |
| 27    | AdSem  | WBKF_         | WB Knowledge Features     | BNois15_S    | Semantic Noise, 150 topics extracted from WeeBit Corpus                        |
| 28    | AdSem  | WBKF_         | WB Knowledge Features     | BTopc15_S    | Number of topics, 150 topics extracted from WeeBit Corpus                      |
| 29    | AdSem  | WBKF_         | WB Knowledge Features     | BRich20_S    | Semantic Richness, 200 topics extracted from WeeBit Corpus                 |
| 30    | AdSem  | WBKF_         | WB Knowledge Features     | BClar20_S    | Semantic Clarity, 200 topics extracted from WeeBit Corpus                      |
| 31    | AdSem  | WBKF_         | WB Knowledge Features     | BNois20_S    | Semantic Noise, 200 topics extracted from WeeBit Corpus                        |
| 32    | AdSem  | WBKF_         | WB Knowledge Features     | BTopc20_S    | Number of topics, 200 topics extracted from WeeBit Corpus                      |
| 33    | AdSem  | OSKF_         | OSE Knowledge Features | ORich05_S    | Semantic Richness, 50 topics extracted from OneStopEng Corpus              |
| 34    | AdSem  | OSKF_         | OSE Knowledge Features | OClar05_S    | Semantic Clarity, 50 topics extracted from OneStopEng Corpus                   |
| 35    | AdSem  | OSKF_         | OSE Knowledge Features | ONois05_S    | Semantic Noise, 50 topics extracted from OneStopEng Corpus                     |
| 36    | AdSem  | OSKF_         | OSE Knowledge Features | OTopc05_S    | Number of topics, 50 topics extracted from OneStopEng Corpus                   |
| 37    | AdSem  | OSKF_         | OSE Knowledge Features | ORich10_S    | Semantic Richness, 100 topics extracted from OneStopEng Corpus             |
| 38    | AdSem  | OSKF_         | OSE Knowledge Features | OClar10_S    | Semantic Clarity, 100 topics extracted from OneStopEng Corpus                  |
| 39    | AdSem  | OSKF_         | OSE Knowledge Features | ONois10_S    | Semantic Noise, 100 topics extracted from OneStopEng Corpus                    |
| 40    | AdSem  | OSKF_         | OSE Knowledge Features | OTopc10_S    | Number of topics, 100 topics extracted from OneStopEng Corpus                  |
| 41    | AdSem  | OSKF_         | OSE Knowledge Features | ORich15_S    | Semantic Richness, 150 topics extracted from OneStopEng Corpus             |
| 42    | AdSem  | OSKF_         | OSE Knowledge Features | OClar15_S    | Semantic Clarity, 150 topics extracted from OneStopEng Corpus                  |
| 43    | AdSem  | OSKF_         | OSE Knowledge Features | ONois15_S    | Semantic Noise, 150 topics extracted from OneStopEng Corpus                    |
| 44    | AdSem  | OSKF_         | OSE Knowledge Features | OTopc15_S    | Number of topics, 150 topics extracted from OneStopEng Corpus                  |
| 45    | AdSem  | OSKF_         | OSE Knowledge Features | ORich20_S    | Semantic Richness, 200 topics extracted from OneStopEng Corpus             |
| 46    | AdSem  | OSKF_         | OSE Knowledge Features | OClar20_S    | Semantic Clarity, 200 topics extracted from OneStopEng Corpus                  |
| 47    | AdSem  | OSKF_         | OSE Knowledge Features | ONois20_S    | Semantic Noise, 200 topics extracted from OneStopEng Corpus                    |
| 48    | AdSem  | OSKF_         | OSE Knowledge Features | OTopc20_S    | Number of topics, 200 topics extracted from OneStopEng Corpus                  |
| 49    | Disco           | EnDF_         | Entity Density Features              | to_EntiM_C   | total number of Entities Mentions counts                                       |
| 50    | Disco           | EnDF_         | Entity Density Features              | as_EntiM_C   | average number of Entities Mentions counts per sentence                        |
| 51    | Disco           | EnDF_         | Entity Density Features              | at_EntiM_C   | average number of Entities Mentions counts per token (word)                    |
| 52    | Disco           | EnDF_         | Entity Density Features              | to_UEnti_C   | total number of unique Entities                                                |
| 53    | Disco           | EnDF_         | Entity Density Features              | as_UEnti_C   | average number of unique Entities per sentence                                 |
| 54    | Disco           | EnDF_         | Entity Density Features              | at_UEnti_C   | average number of unique Entities per token (word)                             |
| 55    | Disco           | EnGF_         | Entity Grid Features                 | ra_SSTo_C   | ratio of ss transitions to total                                               |
| 56    | Disco           | EnGF_         | Entity Grid Features                 | ra_SOTo_C   | ratio of so transitions to total                                               |
| 57    | Disco           | EnGF_         | Entity Grid Features                 | ra_SXTo_C   | ratio of sx transitions to total                                               |
| 58    | Disco           | EnGF_         | Entity Grid Features                 | ra_SNTo_C   | ratio of sn transitions to total                                               |
| 59    | Disco           | EnGF_         | Entity Grid Features                 | ra_OSTo_C   | ratio of os transitions to total                                               |
| 60    | Disco           | EnGF_         | Entity Grid Features                 | ra_OOTo_C   | ratio of oo transitions to total                                               |
| 61    | Disco           | EnGF_         | Entity Grid Features                 | ra_OXTo_C   | ratio of ox transitions to total                                               |
| 62    | Disco           | EnGF_         | Entity Grid Features                 | ra_ONTo_C   | ratio of on transitions to total                                               |
| 63    | Disco           | EnGF_         | Entity Grid Features                 | ra_XSTo_C   | ratio of xs transitions to total                                               |
| 64    | Disco           | EnGF_         | Entity Grid Features                 | ra_XOTo_C   | ratio of xo transitions to total                                               |
| 65    | Disco           | EnGF_         | Entity Grid Features                 | ra_XXTo_C   | ratio of xx transitions to total                                               |
| 66    | Disco           | EnGF_         | Entity Grid Features                 | ra_XNTo_C   | ratio of xn transitions to total                                               |
| 67    | Disco           | EnGF_         | Entity Grid Features                 | ra_NSTo_C   | ratio of ns transitions to total                                               |
| 68    | Disco           | EnGF_         | Entity Grid Features                 | ra_NOTo_C   | ratio of no transitions to total                                               |
| 69    | Disco           | EnGF_         | Entity Grid Features                 | ra_NXTo_C   | ratio of nx transitions to total                                               |
| 70    | Disco           | EnGF_         | Entity Grid Features                 | ra_NNTo_C   | ratio of nn transitions to total                                               |
| 71    | Disco           | EnGF_         | Entity Grid Features                 | LoCohPA_S    | Local Coherence for PA score                                                   |
| 72    | Disco           | EnGF_         | Entity Grid Features                 | LoCohPW_S    | Local Coherence for PW score                                                   |
| 73    | Disco           | EnGF_         | Entity Grid Features                 | LoCohPU_S    | Local Coherence for PU score                                                   |
| 74    | Disco           | EnGF_         | Entity Grid Features                 | LoCoDPA_S    | Local Coherence distance for PA score                                          |
| 75    | Disco           | EnGF_         | Entity Grid Features                 | LoCoDPW_S    | Local Coherence distance for PW score                                          |
| 76    | Disco           | EnGF_         | Entity Grid Features                 | LoCoDPU_S    | Local Coherence distance for PU score                                          |
| 77    | Synta           | PhrF_         | Phrasal Features                     | to_NoPhr_C   | total count of Noun phrases                                                    |
| 78    | Synta           | PhrF_         | Phrasal Features                     | as_NoPhr_C   | average count of Noun phrases per sentence                                     |
| 79    | Synta           | PhrF_         | Phrasal Features                     | at_NoPhr_C   | average count of Noun phrases per token                                        |
| 80    | Synta           | PhrF_         | Phrasal Features                     | ra_NoVeP_C   | ratio of Noun phrases count to Verb phrases count                              |
| 81    | Synta           | PhrF_         | Phrasal Features                     | ra_NoSuP_C   | ratio of Noun phrases count to Subordinate Clauses count                       |
| 82    | Synta           | PhrF_         | Phrasal Features                     | ra_NoPrP_C   | ratio of Noun phrases count to Prep phrases count                              |
| 83    | Synta           | PhrF_         | Phrasal Features                     | ra_NoAjP_C   | ratio of Noun phrases count to Adj phrases count                               |
| 84    | Synta           | PhrF_         | Phrasal Features                     | ra_NoAvP_C   | ratio of Noun phrases count to Adv phrases count                               |
| 85    | Synta           | PhrF_         | Phrasal Features                     | to_VePhr_C   | total count of Verb phrases                                                    |
| 86    | Synta           | PhrF_         | Phrasal Features                     | as_VePhr_C   | average count of Verb phrases per sentence                                     |
| 87    | Synta           | PhrF_         | Phrasal Features                     | at_VePhr_C   | average count of Verb phrases per token                                        |
| 88    | Synta           | PhrF_         | Phrasal Features                     | ra_VeNoP_C   | ratio of Verb phrases count to Noun phrases count                              |
| 89    | Synta           | PhrF_         | Phrasal Features                     | ra_VeSuP_C   | ratio of Verb phrases count to Subordinate Clauses count                       |
| 90    | Synta           | PhrF_         | Phrasal Features                     | ra_VePrP_C   | ratio of Verb phrases count to Prep phrases count                              |
| 91    | Synta           | PhrF_         | Phrasal Features                     | ra_VeAjP_C   | ratio of Verb phrases count to Adj phrases count                               |
| 92    | Synta           | PhrF_         | Phrasal Features                     | ra_VeAvP_C   | ratio of Verb phrases count to Adv phrases count                               |
| 93    | Synta           | PhrF_         | Phrasal Features                     | to_SuPhr_C   | total count of Subordinate Clauses                                             |
| 94    | Synta           | PhrF_         | Phrasal Features                     | as_SuPhr_C   | average count of Subordinate Clauses per sentence                              |
| 95    | Synta           | PhrF_         | Phrasal Features                     | at_SuPhr_C   | average count of Subordinate Clauses per token                                 |
| 96    | Synta           | PhrF_         | Phrasal Features                     | ra_SuNoP_C   | ratio of Subordinate Clauses count to Noun phrases count                       |
| 97    | Synta           | PhrF_         | Phrasal Features                     | ra_SuVeP_C   | ratio of Subordinate Clauses count to Verb phrases count                       |
| 98    | Synta           | PhrF_         | Phrasal Features                     | ra_SuPrP_C   | ratio of Subordinate Clauses count to Prep phrases count                       |
| 99    | Synta           | PhrF_         | Phrasal Features                     | ra_SuAjP_C   | ratio of Subordinate Clauses count to Adj phrases count                        |
| 100   | Synta           | PhrF_         | Phrasal Features                     | ra_SuAvP_C   | ratio of Subordinate Clauses count to Adv phrases count                        |
| 101   | Synta           | PhrF_         | Phrasal Features                     | to_PrPhr_C   | total count of prepositional phrases                                           |
| 102   | Synta           | PhrF_         | Phrasal Features                     | as_PrPhr_C   | average count of prepositional phrases per sentence                            |
| 103   | Synta           | PhrF_         | Phrasal Features                     | at_PrPhr_C   | average count of prepositional phrases per token                               |
| 104   | Synta           | PhrF_         | Phrasal Features                     | ra_PrNoP_C   | ratio of Prep phrases count to Noun phrases count                              |
| 105   | Synta           | PhrF_         | Phrasal Features                     | ra_PrVeP_C   | ratio of Prep phrases count to Verb phrases count                              |
| 106   | Synta           | PhrF_         | Phrasal Features                     | ra_PrSuP_C   | ratio of Prep phrases count to Subordinate Clauses count                       |
| 107   | Synta           | PhrF_         | Phrasal Features                     | ra_PrAjP_C   | ratio of Prep phrases count to Adj phrases count                               |
| 108   | Synta           | PhrF_         | Phrasal Features                     | ra_PrAvP_C   | ratio of Prep phrases count to Adv phrases count                               |
| 109   | Synta           | PhrF_         | Phrasal Features                     | to_AjPhr_C   | total count of Adjective phrases                                               |
| 110   | Synta           | PhrF_         | Phrasal Features                     | as_AjPhr_C   | average count of Adjective phrases per sentence                                |
| 111   | Synta           | PhrF_         | Phrasal Features                     | at_AjPhr_C   | average count of Adjective phrases per token                                   |
| 112   | Synta           | PhrF_         | Phrasal Features                     | ra_AjNoP_C   | ratio of Adj phrases count to Noun phrases count                               |
| 113   | Synta           | PhrF_         | Phrasal Features                     | ra_AjVeP_C   | ratio of Adj phrases count to Verb phrases count                               |
| 114   | Synta           | PhrF_         | Phrasal Features                     | ra_AjSuP_C   | ratio of Adj phrases count to Subordinate Clauses count                        |
| 115   | Synta           | PhrF_         | Phrasal Features                     | ra_AjPrP_C   | ratio of Adj phrases count to Prep phrases count                               |
| 116   | Synta           | PhrF_         | Phrasal Features                     | ra_AjAvP_C   | ratio of Adj phrases count to Adv phrases count                                |
| 117   | Synta           | PhrF_         | Phrasal Features                     | to_AvPhr_C   | total count of Adverb phrases                                                  |
| 118   | Synta           | PhrF_         | Phrasal Features                     | as_AvPhr_C   | average count of Adverb phrases per sentence                                   |
| 119   | Synta           | PhrF_         | Phrasal Features                     | at_AvPhr_C   | average count of Adverb phrases per token                                      |
| 120   | Synta           | PhrF_         | Phrasal Features                     | ra_AvNoP_C   | ratio of Adv phrases count to Noun phrases count                               |
| 121   | Synta           | PhrF_         | Phrasal Features                     | ra_AvVeP_C   | ratio of Adv phrases count to Verb phrases count                               |
| 122   | Synta           | PhrF_         | Phrasal Features                     | ra_AvSuP_C   | ratio of Adv phrases count to Subordinate Clauses count                        |
| 123   | Synta           | PhrF_         | Phrasal Features                     | ra_AvPrP_C   | ratio of Adv phrases count to Prep phrases count                               |
| 124   | Synta           | PhrF_         | Phrasal Features                     | ra_AvAjP_C   | ratio of Adv phrases count to Adj phrases count                                |
| 125   | Synta           | TrSF_         | Tree Structure Features              | to_TreeH_C   | total Tree height of all sentences                                             |
| 126   | Synta           | TrSF_         | Tree Structure Features              | as_TreeH_C   | average Tree height per sentence                                               |
| 127   | Synta           | TrSF_         | Tree Structure Features              | at_TreeH_C   | average Tree height per token (word)                                           |
| 128   | Synta           | TrSF_         | Tree Structure Features              | to_FTree_C   | total length of flattened Trees                                                |
| 129   | Synta           | TrSF_         | Tree Structure Features              | as_FTree_C   | average length of flattened Trees per sentence                                 |
| 130   | Synta           | TrSF_         | Tree Structure Features              | at_FTree_C   | average length of flattened Trees per token (word)                             |
| 131   | Synta           | POSF_         | Part-of-Speech Features              | to_NoTag_C   | total count of Noun POS tags                                                   |
| 132   | Synta           | POSF_         | Part-of-Speech Features              | as_NoTag_C   | average count of Noun POS tags per sentence                                    |
| 133   | Synta           | POSF_         | Part-of-Speech Features              | at_NoTag_C   | average count of Noun POS tags per token                                       |
| 134   | Synta           | POSF_         | Part-of-Speech Features              | ra_NoAjT_C   | ratio of Noun POS count to Adjective POS count                                 |
| 135   | Synta           | POSF_         | Part-of-Speech Features              | ra_NoVeT_C   | ratio of Noun POS count to Verb POS count                                      |
| 136   | Synta           | POSF_         | Part-of-Speech Features              | ra_NoAvT_C   | ratio of Noun POS count to Adverb POS count                                    |
| 137   | Synta           | POSF_         | Part-of-Speech Features              | ra_NoSuT_C   | ratio of Noun POS count to Subordinating Conjunction count                     |
| 138   | Synta           | POSF_         | Part-of-Speech Features              | ra_NoCoT_C   | ratio of Noun POS count to Coordinating Conjunction count                      |
| 139   | Synta           | POSF_         | Part-of-Speech Features              | to_VeTag_C   | total count of Verb POS tags                                                   |
| 140   | Synta           | POSF_         | Part-of-Speech Features              | as_VeTag_C   | average count of Verb POS tags per sentence                                    |
| 141   | Synta           | POSF_         | Part-of-Speech Features              | at_VeTag_C   | average count of Verb POS tags per token                                       |
| 142   | Synta           | POSF_         | Part-of-Speech Features              | ra_VeAjT_C   | ratio of Verb POS count to Adjective POS count                                 |
| 143   | Synta           | POSF_         | Part-of-Speech Features              | ra_VeNoT_C   | ratio of Verb POS count to Noun POS count                                      |
| 144   | Synta           | POSF_         | Part-of-Speech Features              | ra_VeAvT_C   | ratio of Verb POS count to Adverb POS count                                    |
| 145   | Synta           | POSF_         | Part-of-Speech Features              | ra_VeSuT_C   | ratio of Verb POS count to Subordinating Conjunction count                     |
| 146   | Synta           | POSF_         | Part-of-Speech Features              | ra_VeCoT_C   | ratio of Verb POS count to Coordinating Conjunction count                      |
| 147   | Synta           | POSF_         | Part-of-Speech Features              | to_AjTag_C   | total count of Adjective POS tags                                              |
| 148   | Synta           | POSF_         | Part-of-Speech Features              | as_AjTag_C   | average count of Adjective POS tags per sentence                               |
| 149   | Synta           | POSF_         | Part-of-Speech Features              | at_AjTag_C   | average count of Adjective POS tags per token                                  |
| 150   | Synta           | POSF_         | Part-of-Speech Features              | ra_AjNoT_C   | ratio of Adjective POS count to Noun POS count                                 |
| 151   | Synta           | POSF_         | Part-of-Speech Features              | ra_AjVeT_C   | ratio of Adjective POS count to Verb POS count                                 |
| 152   | Synta           | POSF_         | Part-of-Speech Features              | ra_AjAvT_C   | ratio of Adjective POS count to Adverb POS count                               |
| 153   | Synta           | POSF_         | Part-of-Speech Features              | ra_AjSuT_C   | ratio of Adjective POS count to Subordinating Conjunction count                |
| 154   | Synta           | POSF_         | Part-of-Speech Features              | ra_AjCoT_C   | ratio of Adjective POS count to Coordinating Conjunction count                 |
| 155   | Synta           | POSF_         | Part-of-Speech Features              | to_AvTag_C   | total count of Adverb POS tags                                                 |
| 156   | Synta           | POSF_         | Part-of-Speech Features              | as_AvTag_C   | average count of Adverb POS tags per sentence                                  |
| 157   | Synta           | POSF_         | Part-of-Speech Features              | at_AvTag_C   | average count of Adverb POS tags per token                                     |
| 158   | Synta           | POSF_         | Part-of-Speech Features              | ra_AvAjT_C   | ratio of Adverb POS count to Adjective POS count                               |
| 159   | Synta           | POSF_         | Part-of-Speech Features              | ra_AvNoT_C   | ratio of Adverb POS count to Noun POS count                                    |
| 160   | Synta           | POSF_         | Part-of-Speech Features              | ra_AvVeT_C   | ratio of Adverb POS count to Verb POS count                                    |
| 161   | Synta           | POSF_         | Part-of-Speech Features              | ra_AvSuT_C   | ratio of Adverb POS count to Subordinating Conjunction count                   |
| 162   | Synta           | POSF_         | Part-of-Speech Features              | ra_AvCoT_C   | ratio of Adverb POS count to Coordinating Conjunction count                    |
| 163   | Synta           | POSF_         | Part-of-Speech Features              | to_SuTag_C   | total count of Subordinating Conjunction POS tags                              |
| 164   | Synta           | POSF_         | Part-of-Speech Features              | as_SuTag_C   | average count of Subordinating Conjunction POS tags per sentence               |
| 165   | Synta           | POSF_         | Part-of-Speech Features              | at_SuTag_C   | average count of Subordinating Conjunction POS tags per token                  |
| 166   | Synta           | POSF_         | Part-of-Speech Features              | ra_SuAjT_C   | ratio of Subordinating Conjunction POS count to Adjective POS count            |
| 167   | Synta           | POSF_         | Part-of-Speech Features              | ra_SuNoT_C   | ratio of Subordinating Conjunction POS count to Noun POS count                 |
| 168   | Synta           | POSF_         | Part-of-Speech Features              | ra_SuVeT_C   | ratio of Subordinating Conjunction POS count to Verb POS count                 |
| 169   | Synta           | POSF_         | Part-of-Speech Features              | ra_SuAvT_C   | ratio of Subordinating Conjunction POS count to Adverb POS count               |
| 170   | Synta           | POSF_         | Part-of-Speech Features              | ra_SuCoT_C   | ratio of Subordinating Conjunction POS count to Coordinating Conjunction count |
| 171   | Synta           | POSF_         | Part-of-Speech Features              | to_CoTag_C   | total count of Coordinating Conjunction POS tags                               |
| 172   | Synta           | POSF_         | Part-of-Speech Features              | as_CoTag_C   | average count of Coordinating Conjunction POS tags per sentence                |
| 173   | Synta           | POSF_         | Part-of-Speech Features              | at_CoTag_C   | average count of Coordinating Conjunction POS tags per token                   |
| 174   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoAjT_C   | ratio of Coordinating Conjunction POS count to Adjective POS count             |
| 175   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoNoT_C   | ratio of Coordinating Conjunction POS count to Noun POS count                  |
| 176   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoVeT_C   | ratio of Coordinating Conjunction POS count to Verb POS count                  |
| 177   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoAvT_C   | ratio of Coordinating Conjunction POS count to Adverb POS count                |
| 178   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoSuT_C   | ratio of Coordinating Conjunction POS count to Subordinating Conjunction count |
| 179   | Synta           | POSF_         | Part-of-Speech Features              | to_ContW_C   | total count of Content words                                                   |
| 180   | Synta           | POSF_         | Part-of-Speech Features              | as_ContW_C   | average count of Content words per sentence                                    |
| 181   | Synta           | POSF_         | Part-of-Speech Features              | at_ContW_C   | average count of Content words per token                                       |
| 182   | Synta           | POSF_         | Part-of-Speech Features              | to_FuncW_C   | total count of Function words                                                  |
| 183   | Synta           | POSF_         | Part-of-Speech Features              | as_FuncW_C   | average count of Function words per sentence                                   |
| 184   | Synta           | POSF_         | Part-of-Speech Features              | at_FuncW_C   | average count of Function words per token                                      |
| 185   | Synta           | POSF_         | Part-of-Speech Features              | ra_CoFuW_C   | ratio of Content words to Function words                                       |
| 186   | LxSem     | VarF_         | Variation Ratio Features             | SimpNoV_S    | unique Nouns/total Nouns (Noun Variation-1)                                    |
| 187   | LxSem     | VarF_         | Variation Ratio Features             | SquaNoV_S    | (unique Nouns**2)/total Nouns (Squared Noun Variation-1)                       |
| 188   | LxSem     | VarF_         | Variation Ratio Features             | CorrNoV_S    | unique Nouns/sqrt(2*total Nouns) (Corrected Noun Variation-1)                  |
| 189   | LxSem     | VarF_         | Variation Ratio Features             | SimpVeV_S    | unique Verbs/total Verbs (Verb Variation-1)                                    |
| 190   | LxSem     | VarF_         | Variation Ratio Features             | SquaVeV_S    | (unique Verbs**2)/total Verbs (Squared Verb Variation-1)                       |
| 191   | LxSem     | VarF_         | Variation Ratio Features             | CorrVeV_S    | unique Verbs/sqrt(2*total Verbs) (Corrected Verb Variation-1)                  |
| 192   | LxSem     | VarF_         | Variation Ratio Features             | SimpAjV_S    | unique Adjectives/total Adjectives (Adjective Variation-1)                     |
| 193   | LxSem     | VarF_         | Variation Ratio Features             | SquaAjV_S    | (unique Adjectives**2)/total Adjectives (Squared Adjective Variation-1)        |
| 194   | LxSem     | VarF_         | Variation Ratio Features             | CorrAjV_S    | unique Adjectives/sqrt(2*total Adjectives) (Corrected Adjective Variation-1)   |
| 195   | LxSem     | VarF_         | Variation Ratio Features             | SimpAvV_S    | unique Adverbs/total Adverbs (AdVerb Variation-1)                              |
| 196   | LxSem     | VarF_         | Variation Ratio Features             | SquaAvV_S    | (unique Adverbs**2)/total Adverbs (Squared AdVerb Variation-1)                 |
| 197   | LxSem     | VarF_         | Variation Ratio Features             | CorrAvV_S    | unique Adverbs/sqrt(2*total Adverbs) (Corrected AdVerb Variation-1)            |
| 198   | LxSem     | TTRF_         | Type Token Ratio Features            | SimpTTR_S    | unique tokens/total tokens (TTR)                                               |
| 199   | LxSem     | TTRF_         | Type Token Ratio Features            | CorrTTR_S    | unique tokens/sqrt(2*total tokens) (Corrected TTR)                             |
| 200   | LxSem     | TTRF_         | Type Token Ratio Features            | BiLoTTR_S    | log(unique tokens)/log(total tokens) (Bi-Logarithmic TTR)                      |
| 201   | LxSem     | TTRF_         | Type Token Ratio Features            | UberTTR_S    | (log(unique tokens))^2/log(total tokens/unique tokens) (Uber Index)            |
| 202   | LxSem     | TTRF_         | Type Token Ratio Features            | MTLDTTR_S    | Measure of Textual Lexical Diversity (default TTR = 0.72)                      |
| 203   | LxSem     | PsyF_         | Psycholinguistic Features            | to_AAKuW_C   | total AoA (Age of Acquisition) of words                                        |
| 204   | LxSem     | PsyF_         | Psycholinguistic Features            | as_AAKuW_C   | average AoA of words per sentence                                              |
| 205   | LxSem     | PsyF_         | Psycholinguistic Features            | at_AAKuW_C   | average AoA of words per token                                                 |
| 206   | LxSem     | PsyF_         | Psycholinguistic Features            | to_AAKuL_C   | total lemmas AoA of lemmas                                                     |
| 207   | LxSem     | PsyF_         | Psycholinguistic Features            | as_AAKuL_C   | average lemmas AoA of lemmas per sentence                                      |
| 208   | LxSem     | PsyF_         | Psycholinguistic Features            | at_AAKuL_C   | average lemmas AoA of lemmas per token                                         |
| 209   | LxSem     | PsyF_         | Psycholinguistic Features            | to_AABiL_C   | total lemmas AoA of lemmas, Bird norm                                          |
| 210   | LxSem     | PsyF_         | Psycholinguistic Features            | as_AABiL_C   | average lemmas AoA of lemmas, Bird norm per sentence                           |
| 211   | LxSem     | PsyF_         | Psycholinguistic Features            | at_AABiL_C   | average lemmas AoA of lemmas, Bird norm per token                              |
| 212   | LxSem     | PsyF_         | Psycholinguistic Features            | to_AABrL_C   | total lemmas AoA of lemmas, Bristol norm                                       |
| 213   | LxSem     | PsyF_         | Psycholinguistic Features            | as_AABrL_C   | average lemmas AoA of lemmas, Bristol norm per sentence                        |
| 214   | LxSem     | PsyF_         | Psycholinguistic Features            | at_AABrL_C   | average lemmas AoA of lemmas, Bristol norm per token                           |
| 215   | LxSem     | PsyF_         | Psycholinguistic Features            | to_AACoL_C   | total AoA of lemmas, Cortese and Khanna norm                                   |
| 216   | LxSem     | PsyF_         | Psycholinguistic Features            | as_AACoL_C   | average AoA of lemmas, Cortese and Khanna norm per sentence                    |
| 217   | LxSem     | PsyF_         | Psycholinguistic Features            | at_AACoL_C   | average AoA of lemmas, Cortese and Khanna norm per token                       |
| 218   | LxSem     | WorF_         | Word Familiarity              | to_SbFrQ_C   | total SubtlexUS FREQcount value                                                |
| 219   | LxSem     | WorF_         | Word Familiarity              | as_SbFrQ_C   | average SubtlexUS FREQcount value per sentenc                                  |
| 220   | LxSem     | WorF_         | Word Familiarity              | at_SbFrQ_C   | average SubtlexUS FREQcount value per token                                    |
| 221   | LxSem     | WorF_         | Word Familiarity              | to_SbCDC_C   | total SubtlexUS CDcount value                                                  |
| 222   | LxSem     | WorF_         | Word Familiarity              | as_SbCDC_C   | average SubtlexUS CDcount value per sentence                                   |
| 223   | LxSem     | WorF_         | Word Familiarity              | at_SbCDC_C   | average SubtlexUS CDcount value per token                                      |
| 224   | LxSem     | WorF_         | Word Familiarity              | to_SbFrL_C   | total SubtlexUS FREQlow value                                                  |
| 225   | LxSem     | WorF_         | Word Familiarity              | as_SbFrL_C   | average SubtlexUS FREQlow value per sentence                                   |
| 226   | LxSem     | WorF_         | Word Familiarity              | at_SbFrL_C   | average SubtlexUS FREQlow value per token                                      |
| 227   | LxSem     | WorF_         | Word Familiarity              | to_SbCDL_C   | total SubtlexUS CDlow value                                                    |
| 228   | LxSem     | WorF_         | Word Familiarity              | as_SbCDL_C   | average SubtlexUS CDlow value per sentence                                     |
| 229   | LxSem     | WorF_         | Word Familiarity              | at_SbCDL_C   | average SubtlexUS CDlow value per token                                        |
| 230   | LxSem     | WorF_         | Word Familiarity              | to_SbSBW_C   | total SubtlexUS SUBTLWF value                                                  |
| 231   | LxSem     | WorF_         | Word Familiarity              | as_SbSBW_C   | average SubtlexUS SUBTLWF value per sentence                                   |
| 232   | LxSem     | WorF_         | Word Familiarity              | at_SbSBW_C   | average SubtlexUS SUBTLWF value per token                                      |
| 233   | LxSem     | WorF_         | Word Familiarity              | to_SbL1W_C   | total SubtlexUS Lg10WF value                                                   |
| 234   | LxSem     | WorF_         | Word Familiarity              | as_SbL1W_C   | average SubtlexUS Lg10WF value per sentence                                    |
| 235   | LxSem     | WorF_         | Word Familiarity              | at_SbL1W_C   | average SubtlexUS Lg10WF value per token                                       |
| 236   | LxSem     | WorF_         | Word Familiarity              | to_SbSBC_C   | total SubtlexUS SUBTLCD value                                                  |
| 237   | LxSem     | WorF_         | Word Familiarity              | as_SbSBC_C   | average SubtlexUS SUBTLCD value per sentence                                   |
| 238   | LxSem     | WorF_         | Word Familiarity              | at_SbSBC_C   | average SubtlexUS SUBTLCD value per token                                      |
| 239   | LxSem     | WorF_         | Word Familiarity              | to_SbL1C_C   | total SubtlexUS Lg10CD value                                                   |
| 240   | LxSem     | WorF_         | Word Familiarity              | as_SbL1C_C   | average SubtlexUS Lg10CD value per sentence                                    |
| 241   | LxSem     | WorF_         | Word Familiarity              | at_SbL1C_C   | average SubtlexUS Lg10CD value per token                                       |
| 242   | ShaTr     | ShaF_         | Shallow Features                     | TokSenM_S   | total count of tokens x total count of sentence                                |
| 243   | ShaTr     | ShaF_         | Shallow Features                     | TokSenS_S   | sqrt(total count of tokens x total count of sentence)                                |
| 244   | ShaTr     | ShaF_         | Shallow Features                     | TokSenL_S   | log(total count of tokens)/log(total count of sentence)                           |
| 245   | ShaTr     | ShaF_         | Shallow Features                     | as_Token_C   | average count of tokens per sentence                                           |
| 246   | ShaTr     | ShaF_         | Shallow Features                     | as_Sylla_C   | average count of syllables per sentence                                        |
| 247   | ShaTr     | ShaF_         | Shallow Features                     | at_Sylla_C         | average count of syllables per token                                           |
| 248   | ShaTr     | ShaF_         | Shallow Features                     | as_Chara_C   | average count of characters per sentence                                       |
| 249   | ShaTr     | ShaF_         | Shallow Features                     | at_Chara_C   | average count of characters per token                                          |
| 250   | ShaTr     | TraF_         | Traditional Formulas         | SmogInd_S    | Smog Index                                                                     |
| 251   | ShaTr     | TraF_         | Traditional Formulas         | ColeLia_S    | Coleman Liau Readability Score                                                 |
| 252   | ShaTr     | TraF_         | Traditional Formulas         | Gunning_S    | Gunning Fog Count Score                                                                    |
| 253   | ShaTr     | TraF_         | Traditional Formulas         | AutoRea_S    | New Automated Readability Index                                                    |
| 254   | ShaTr     | TraF_         | Traditional Formulas         | FleschG_S    | Flesch Kincaid Grade Level                                                           |
| 255   | ShaTr     | TraF_         | Traditional Formulas         | LinseaW_S    | Linsear Write Formula Score"""

lsca_names = lca_names + sca_names
name_map = {lsca_names[i]: full_names[i] for i in range(len(lsca_names))}

type_map = {lingfeat_names[i]: lingfeat_subtypes[i] for i in range(len(lingfeat_names))}
type_map.update({n: 'lexical' for n in lca_names})
type_map.update({n: 'syntax' for n in sca_names})


# from lingfeat_full_names import lf_names
lf_names = lf_names.split('\n')

lf_names = [tuple(x.split('|')[5:7]) for x in lf_names]
lf_map = {k.strip(): v.strip() for k,v in lf_names}
name_map.update(lf_map)

used_indices = [
        1, 2, 3, 4, 5, 6, 7, 10, 11, 18, 25, 30, 31, 34, 35, 36, 37, 39, 40, 41, 57,
        63, 64, 65, 66, 67, 68, 73, 121, 124, 129, 134, 136, 254,
        257, 258, 261, 263, 272, 274 
        ]

eval_indices = [4,5,6,18,257,272]
eval_indices = [used_indices.index(idx) for idx in eval_indices]

lftk_df = pd.read_csv('lftk_ids.csv')

lftk_types = {row['key']: row['domain'] for i,row in lftk_df.iterrows()}
type_map.update(lftk_types)

type_map = {k:\
        'syntax' if v == 'surface'\
        else 'lexical' if v == 'lexico-semantics'\
        else v\
        for k,v in type_map.items()}

lftkplus_names = lca_names + sca_names + lftk_names
lftkplus_names = [lftkplus_names[i] for i in used_indices]

lftk_map = {k: v for k,v in zip(lftk_names, lftk_full_names)}
name_map.update(lftk_map)
rev_name_map = {v: k for k,v in name_map.items()}
