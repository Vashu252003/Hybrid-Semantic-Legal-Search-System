# Corpus Notes

`legal_cases.json` is a source-backed validation corpus. It contains imported Indian judgment records plus official law/source records imported from the source registry.

The judgment portion can be rebuilt from an actual public Indian legal judgment dataset using `scripts/import_hf_legal_dataset.py`. By default, that importer rebuilds `legal_cases.json` from actual dataset records only. The old curated statutory/reference records are kept only in `legal_cases_curated_backup.json` for transparency and optional controlled benchmarking.

Official law/source records can be imported with `scripts/import_india_code_statutes.py`. The importer reads `data/india_code_sources.json`, fetches official India Code/regulator/government pages, discovers official PDFs where available, extracts source text, and appends searchable `IN-LAW-*` records to `legal_cases.json`.

Manual applicable-law guidance is disabled. `data/applicable_laws.json` is intentionally empty; the application relies on retrieved judgment and law/source records instead of curated law mappings.

Actual judgment dataset source:

- aadityaJagdale/cleaned_legal_dataset on Hugging Face: https://huggingface.co/datasets/aadityaJagdale/cleaned_legal_dataset
- Dataset card states it contains processed Indian legal judgments with fields such as petitioner, respondent, date, citations, judgement text, and tags, sourced from Supreme Court of India, High Courts, and judicial information systems.

Reference points used while preparing the metadata:

- Kesavananda Bharati v. State of Kerala, decided 24 April 1973: https://en.wikipedia.org/wiki/Kesavananda_Bharati_v._State_of_Kerala
- Maneka Gandhi v. Union of India, decided 25 January 1978: https://iritmold.indianrailways.gov.in/instt/uploads/files/1436779312324-Maneka%20Gandhi%20v.%20Union%20of%20India.pdf
- Vishaka v. State of Rajasthan, decided 13 August 1997: https://www.law.cornell.edu/sites/www.law.cornell.edu/files/women-and-justice/India-Vishaka_-_Ors_vs_State_Of_Rajasthan_-_Ors-1997.pdf
- Justice K.S. Puttaswamy v. Union of India, decided 24 August 2017: https://righttoinformation.wiki/important-decisions/k-s-puttaswamy-vs-union-of-india
- Model Tenancy Act, 2021 overview: https://www.pib.gov.in/Pressreleaseshare.aspx?PRID=1779681
- Consumer Protection Act, 2019: https://www.indiacode.nic.in/handle/123456789/15256
- IRDAI policyholder protection regulations: https://irdai.gov.in/document-detail?documentId=385593
- Copyright Act, 1957: https://www.copyright.gov.in/Copyright_Act_1957/index.html
- Bharatiya Nyaya Sanhita, 2023 official information: https://www.mha.gov.in/en/commoncontent/new-criminal-laws
- Bharatiya Nyaya Sanhita, 2023 on India Code: https://www.indiacode.nic.in/handle/123456789/20062
- Information Technology Act, 2000: https://www.indiacode.nic.in/handle/123456789/1999
- Information Technology Act, 2000 on India Code: https://www.indiacode.nic.in/handle/123456789/15442
- Indian Contract Act, 1872 on India Code: https://www.indiacode.nic.in/handle/123456789/2187?locale=en
- Insurance Act, 1938 on India Code: https://www.indiacode.nic.in/handle/123456789/2304?locale=en
- Copyright Act, 1957 on India Code: https://www.indiacode.nic.in/handle/123456789/1367?view_type=browse
- Bharatiya Nagarik Suraksha Sanhita, 2023 on India Code: https://www.indiacode.nic.in/handle/123456789/21615
- Transfer of Property Act, 1882 on India Code: https://www.indiacode.nic.in/handle/123456789/2338
- Registration Act, 1908 on India Code: https://www.indiacode.nic.in/handle/123456789/2190?locale=en
- Specific Relief Act, 1963 on India Code: https://www.indiacode.nic.in/bitstream/123456789/1583/7/A1963-47.pdf
- Land Acquisition, Rehabilitation and Resettlement Act, 2013 on India Code: https://www.indiacode.nic.in/handle/123456789/2121
