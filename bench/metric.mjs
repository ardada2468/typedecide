/**
 * Mean family balanced accuracy, as openjev's metric contract defines it: within a
 * family, take the recall of each gold class and average them, then average over
 * families. Balanced rather than plain accuracy because the fixture's classes are
 * uneven and a majority-class answerer should not score well.
 */

export function meanFamilyBalancedAccuracy(records) {
  const families = [...new Set(records.map((r) => r.family))].sort();
  if (!families.length) throw new Error('No records to score');

  const per = {};
  for (const family of families) {
    const rows = records.filter((r) => r.family === family);
    const classes = [...new Set(rows.map((r) => r.gold))].sort();
    const recalls = classes.map((c) => {
      const of = rows.filter((r) => r.gold === c);
      return of.filter((r) => r.pred === c).length / of.length;
    });
    per[family] = recalls.reduce((a, b) => a + b, 0) / recalls.length;
  }
  return { per, mean: Object.values(per).reduce((a, b) => a + b, 0) / families.length };
}
