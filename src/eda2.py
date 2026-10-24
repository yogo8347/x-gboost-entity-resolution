import polars as pl, re
from anyascii import anyascii
D="dataset/train/"
rd=lambda f: pl.read_csv(D+f,separator="\t",quote_char=None,infer_schema=False)
gt=rd("train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null("").str.split(",")).explode("matched_entity_ids").filter(pl.col("matched_entity_ids")!="")
gt=gt.rename({"matched_entity_ids":"entity_id","source1_entity_id":"s1"})
s1=rd("train_source1.tsv"); s2=rd("train_source2.tsv")
def norm(c): return pl.col(c).map_elements(lambda s: " ".join(re.findall(r"[a-z0-9]+",anyascii(s).lower())),return_dtype=pl.Utf8)
s2=s2.sample(200000,seed=0).join(gt,on="entity_id",how="left")
s2=s2.with_columns(norm("business_name").alias("nn"))
s1n=s1.with_columns(norm("business_name").alias("nn")).select("entity_id","nn","business_address")
names=set(s1n["nn"].to_list())
s2=s2.with_columns(pl.col("nn").map_elements(lambda x: x in names,return_dtype=pl.Boolean).alias("exact_in_s1"))
print(s2.group_by(pl.col("s1").is_null().alias("unmatched")).agg(pl.col("exact_in_s1").mean(),pl.len()))
un=s2.filter(pl.col("s1").is_null())
print(un["country"].value_counts())
for r in un.filter(pl.col("exact_in_s1")).head(8).iter_rows(named=True):
    print("\nUNMATCHED",r["business_name"],"|",r["business_address"])
    for x in s1n.filter(pl.col("nn")==r["nn"]).head(3).iter_rows(named=True): print("   S1:",x["business_address"])
print("\nrandom unmatched:"); 
for r in un.sample(12,seed=3).iter_rows(named=True): print("  ",r["business_name"],"|",r["business_address"],"|",r["country"])
# name dup rate within S1
print("S1 dup normalized names frac", 1-s1n["nn"].n_unique()/s1n.height)
