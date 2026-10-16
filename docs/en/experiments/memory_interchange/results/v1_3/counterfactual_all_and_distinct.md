> English edition · [Frozen original](../../../../../../experiments/memory_interchange/results/v1_3/counterfactual_all_and_distinct.md) · [English index](../../../../README.md)
>
> Dates, status, execution scope and conclusions describe the original version. Commands, identifiers, data and figures retain their original definitions.

# Current-content and initial-input counterfactuals: all queries and distinguishable subsets

Counts n below include repeated directions/queries sharing states and orbits, not independent experiments. Current target is F_b(q), initial-input target F_b(G_b(p_a)). Matched and wrong-content treatments share one pre-patch mask.

| Seed | Same/cross | mask | All queries n | All-query current agreement | All-query initial agreement | Answer-distinguishable n | Subset current agreement | Subset initial agreement |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 101 | Same | all | 1620864 | 98.2923% | 34.1312% | 1069056 | 97.9984% | 0.7198% |
| 101 | Same | rec | 1593184 | 98.3623% | 34.3655% | 1047694 | 98.0288% | 0.7117% |
| 101 | Same | both | 1567092 | 100.0000% | 34.4619% | 1027042 | 100.0000% | 0.0000% |
| 101 | Same | three | 1542129 | 100.0000% | 34.5787% | 1008881 | 100.0000% | 0.0000% |
| 101 | Cross | all | 1823472 | 45.3743% | 35.6016% | 1069056 | 42.7823% | 26.1131% |
| 101 | Cross | rec | 1792332 | 45.8218% | 35.8194% | 1047694 | 43.2518% | 26.1405% |
| 101 | Cross | both | 1761854 | 46.0252% | 35.8999% | 1027273 | 43.4958% | 26.1302% |
| 101 | Cross | three | 1734030 | 46.3423% | 36.0318% | 1008978 | 43.8398% | 26.1202% |
| 102 | Same | all | 1620864 | 98.3653% | 34.0683% | 1069056 | 98.1827% | 0.6979% |
| 102 | Same | rec | 1594368 | 98.4788% | 34.2117% | 1049848 | 98.2638% | 0.6637% |
| 102 | Same | both | 1570114 | 100.0000% | 34.2964% | 1031621 | 100.0000% | 0.0000% |
| 102 | Same | three | 1545605 | 100.0000% | 34.3165% | 1015208 | 100.0000% | 0.0000% |
| 102 | Cross | all | 1823472 | 45.3521% | 35.4809% | 1069056 | 42.7733% | 25.9362% |
| 102 | Cross | rec | 1793664 | 45.7218% | 35.5830% | 1049848 | 43.1702% | 25.8481% |
| 102 | Cross | both | 1764542 | 45.8808% | 35.6720% | 1031620 | 43.3479% | 25.8863% |
| 102 | Cross | three | 1737206 | 46.1007% | 35.6916% | 1015485 | 43.6095% | 25.8025% |
| 103 | Same | all | 1620864 | 98.1822% | 34.0943% | 1069056 | 97.9634% | 0.7957% |
| 103 | Same | rec | 1591400 | 98.2816% | 34.2980% | 1047076 | 98.0147% | 0.7691% |
| 103 | Same | both | 1564054 | 100.0000% | 34.3828% | 1026288 | 100.0000% | 0.0000% |
| 103 | Same | three | 1537409 | 100.0000% | 34.4822% | 1007276 | 100.0000% | 0.0000% |
| 103 | Cross | all | 1823472 | 45.2593% | 35.5236% | 1069056 | 42.7064% | 26.1003% |
| 103 | Cross | rec | 1790325 | 45.6836% | 35.6630% | 1047076 | 43.1700% | 26.0365% |
| 103 | Cross | both | 1758012 | 45.9719% | 35.7483% | 1026011 | 43.4722% | 25.9546% |
| 103 | Cross | three | 1728101 | 46.2581% | 35.8671% | 1006951 | 43.7726% | 25.9399% |
| 104 | Same | all | 1620864 | 97.8614% | 34.1043% | 1069056 | 97.5915% | 0.9254% |
| 104 | Same | rec | 1586200 | 98.0185% | 34.2989% | 1043770 | 97.7013% | 0.8676% |
| 104 | Same | both | 1554770 | 100.0000% | 34.4098% | 1019777 | 100.0000% | 0.0000% |
| 104 | Same | three | 1523220 | 100.0000% | 34.5189% | 997421 | 100.0000% | 0.0000% |
| 104 | Cross | all | 1823472 | 44.8610% | 35.4451% | 1069056 | 42.2785% | 26.2180% |
| 104 | Cross | rec | 1784475 | 45.2143% | 35.5951% | 1043770 | 42.6483% | 26.2028% |
| 104 | Cross | both | 1746998 | 45.8075% | 35.7727% | 1019678 | 43.2704% | 26.0780% |
| 104 | Cross | three | 1711492 | 46.2058% | 35.8878% | 997307 | 43.6831% | 25.9763% |
| 105 | Same | all | 1620864 | 97.9636% | 34.0807% | 1069056 | 97.7466% | 0.8898% |
| 105 | Same | rec | 1587856 | 98.0871% | 34.2996% | 1044701 | 97.8113% | 0.8598% |
| 105 | Same | both | 1557482 | 100.0000% | 34.3918% | 1021836 | 100.0000% | 0.0000% |
| 105 | Same | three | 1527367 | 100.0000% | 34.5163% | 1000177 | 100.0000% | 0.0000% |
| 105 | Cross | all | 1823472 | 45.1799% | 35.4850% | 1069056 | 42.6141% | 26.0778% |
| 105 | Cross | rec | 1786338 | 45.5802% | 35.6396% | 1044701 | 43.0183% | 26.0207% |
| 105 | Cross | both | 1750248 | 45.9221% | 35.7648% | 1021552 | 43.3716% | 25.9688% |
| 105 | Cross | three | 1716398 | 46.2477% | 35.9015% | 999806 | 43.7029% | 25.9411% |
| 106 | Same | all | 1620864 | 98.3443% | 34.1415% | 1069056 | 98.0844% | 0.7424% |
| 106 | Same | rec | 1594040 | 98.3979% | 34.3420% | 1048614 | 98.1078% | 0.7338% |
| 106 | Same | both | 1568514 | 99.9992% | 34.4105% | 1028777 | 99.9995% | 0.0004% |
| 106 | Same | three | 1547497 | 99.9992% | 34.5418% | 1012960 | 99.9995% | 0.0004% |
| 106 | Cross | all | 1823472 | 45.2221% | 35.5554% | 1069056 | 42.6167% | 26.1282% |
| 106 | Cross | rec | 1793295 | 45.6107% | 35.7275% | 1048614 | 43.0115% | 26.1097% |
| 106 | Cross | both | 1763794 | 45.9067% | 35.7889% | 1028739 | 43.3558% | 26.0086% |
| 106 | Cross | three | 1740264 | 46.1734% | 35.9444% | 1013014 | 43.6321% | 26.0596% |

## Full mappings reconstructed jointly from all 14 queries

| Seed | Same/cross | Repeated mappings n | Candidate full-mapping overlap | Current full-mapping agreement | Initial full-mapping agreement | Distinguishable mappings n | Subset current full-mapping agreement | Subset initial full-mapping agreement |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 101 | Same | 115776 | 15.5887% | 83.7203% | 13.9321% | 97728 | 82.6764% | 0.0000% |
| 101 | Cross | 130248 | 24.9678% | 31.1291% | 16.0049% | 97728 | 30.9747% | 10.8178% |
| 102 | Same | 115776 | 15.5887% | 82.4350% | 13.5339% | 97728 | 81.6255% | 0.0000% |
| 102 | Cross | 130248 | 24.9678% | 30.9011% | 15.3146% | 97728 | 30.7599% | 9.9869% |
| 103 | Same | 115776 | 15.5887% | 80.8527% | 13.5797% | 97728 | 79.6967% | 0.0000% |
| 103 | Cross | 130248 | 24.9678% | 30.6730% | 15.2555% | 97728 | 30.5593% | 10.0115% |
| 104 | Same | 115776 | 15.5887% | 77.2526% | 12.9241% | 97728 | 76.2085% | 0.0000% |
| 104 | Cross | 130248 | 24.9678% | 29.7955% | 14.6897% | 97728 | 29.6333% | 9.5009% |
| 105 | Same | 115776 | 15.5887% | 79.3187% | 13.2895% | 97728 | 78.2232% | 0.0000% |
| 105 | Cross | 130248 | 24.9678% | 30.1064% | 15.1603% | 97728 | 29.8359% | 9.9163% |
| 106 | Same | 115776 | 15.5887% | 87.0094% | 14.4529% | 97728 | 85.9559% | 0.0000% |
| 106 | Cross | 130248 | 24.9678% | 31.4677% | 16.4532% | 97728 | 31.2971% | 11.2864% |

Equality of wrong-content-all counterfactual accuracy and matched-all accuracy is expected from fixed cross-orbit bijective reordering and content-independent c; it is not independent replication. Distinguishable-answer subsets and shared-mask conditional results are retained separately to avoid accidental agreement or treatment-specific selection.
