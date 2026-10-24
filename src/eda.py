import polars as pl
D="dataset/train/"
rd=lambda f: pl.read_csv(D+f,separator="\t",quote_char=None,infer_schema=False)
gt=rd("train_ground_truth.tsv").with_columns(pl.col("matched_entity_ids").fill_null(""))
gt=gt.with_columns(pl.col("matched_entity_ids").str.split(",").list.eval(pl.element().filter(pl.element()!="")).alias("m"))
gt=gt.with_columns(pl.col("m").list.len().alias("n"),
  pl.col("m").list.eval(pl.element().str.starts_with("S2").cast(pl.Int32)).list.sum().alias("n2"))
gt=gt.with_columns((pl.col("n")-pl.col("n2")).alias("n3"))
print("S1 rows",gt.height,"singleton frac",(gt["n"]==0).mean())
print(gt["n"].value_counts().sort("n").head(15))
print(gt.group_by("n2","n3").len().sort("len",descending=True).head(12))
s1=rd("train_source1.tsv"); print(s1["country"].value_counts())
ex=gt.explode("m").drop_nulls("m")
print("total matched ids",ex.height,"unique",ex["m"].n_unique())
s2=rd("train_source2.tsv"); s3=rd("train_source3.tsv")
print("S2",s2.height,"S3",s3.height,"frac S2 matched",ex.filter(pl.col("m").str.starts_with("S2"))["m"].n_unique()/s2.height,
      "frac S3 matched",ex.filter(pl.col("m").str.starts_with("S3"))["m"].n_unique()/s3.height)
for s in (s1,s2,s3): print(s.null_count())
# examples
s=pl.concat([s2,s3])
smp=gt.filter(pl.col("n")>0).sample(6,seed=1)
for r in smp.iter_rows(named=True):
    print("\n#",s1.filter(pl.col("entity_id")==r["source1_entity_id"]).row(0))
    for m in r["m"]: print("   ",s.filter(pl.col("entity_id")==m).row(0))
