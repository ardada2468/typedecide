import test from 'node:test';
import assert from 'node:assert/strict';
import { meanFamilyBalancedAccuracy } from './metric.mjs';

const row = (family, gold, pred) => ({ family, gold, pred });

test('perfect predictions score 1', () => {
  const { mean } = meanFamilyBalancedAccuracy([
    row('f', 'a', 'a'), row('f', 'b', 'b'), row('f', 'c', 'c'),
  ]);
  assert.equal(mean, 1);
});

test('balanced accuracy punishes a majority-class answerer', () => {
  // Nine rows, seven of class a. Always answering "a" gets 78% plain accuracy
  // but only 50% balanced, which is the whole reason we use it.
  const records = [
    ...Array.from({ length: 7 }, () => row('f', 'a', 'a')),
    row('f', 'b', 'a'), row('f', 'b', 'a'),
  ];
  const { mean } = meanFamilyBalancedAccuracy(records);
  assert.equal(mean, 0.5);
});

test('families are averaged evenly regardless of their size', () => {
  const records = [
    row('small', 'a', 'a'),                                     // family scores 1
    ...Array.from({ length: 20 }, () => row('big', 'a', 'b')),  // family scores 0
  ];
  const { mean, per } = meanFamilyBalancedAccuracy(records);
  assert.equal(per.small, 1);
  assert.equal(per.big, 0);
  assert.equal(mean, 0.5);
});

test('a class the model never predicts contributes zero recall', () => {
  const { mean } = meanFamilyBalancedAccuracy([
    row('f', 'a', 'a'), row('f', 'a', 'a'), row('f', 'b', 'a'),
  ]);
  assert.equal(mean, 0.5);
});

test('empty input is refused rather than returning NaN', () => {
  assert.throws(() => meanFamilyBalancedAccuracy([]), /No records/);
});
