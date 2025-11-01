//заперечення
function fuzzyNOT(a) {
  return 1 - a;
}

//кон'юнкція
function fuzzyAND(a, b) {
  return Math.min(a, b);
}

//диз'юнкція
function fuzzyOR(a, b) {
  return Math.max(a, b);
}

//імплікація
function fuzzyIMP(a, b) {
  return Math.max(1 - a, b);
}
//еквівалентність
function fuzzyEQU(a, b) {
  const imp1 = Math.max(1 - a, b); // Імплікація a => b
  const imp2 = Math.max(1 - b, a); // Імплікація b => a
  return Math.min(imp1, imp2); // Кон'юнкція результатів
}

function calculateF9(a, b, c) {
  // Крок 1: Обчислити кон'юнкцію (B ∧ C)
  const conjunction_BC = fuzzyAND(b, c);
  console.log(`Результат (B ∧ C) = min(${b}, ${c}) = ${conjunction_BC}`);

  // Крок 2: Обчислити заперечення результату ¬(B ∧ C)
  const negation_of_BC = fuzzyNOT(conjunction_BC);
  console.log(`Результат ¬(B ∧ C) = 1 - ${conjunction_BC} = ${negation_of_BC}`);

  // Крок 3: Обчислити імплікацію A → ¬(B ∧ C)
  const finalResult = fuzzyIMP(a, negation_of_BC);
  console.log(
    `Результат A → ¬(B ∧ C) = max(1 - ${a}, ${negation_of_BC}) = ${finalResult}`
  );

  return finalResult;
}

const truthValueA = 0.8; // A є істинним на 80%
const truthValueB = 0.6; // B є істинним на 60%
const truthValueC = 0.3; // C є істинним на 30%

console.log(
  `Вхідні дані: A=${truthValueA}, B=${truthValueB}, C=${truthValueC}\n`
);

// Викликаємо функцію для обчислення
const result = calculateF9(truthValueA, truthValueB, truthValueC);

console.log(`\nКінцевий ступінь істинності для f₉: ${result}`);

const testData = [
  { A: 0.1, B: 0.2, C: 0.3 },
  { A: 0.9, B: 0.8, C: 0.7 },
  { A: 0.5, B: 0.5, C: 0.5 },
  { A: 1.0, B: 1.0, C: 1.0 },
  { A: 0.0, B: 0.0, C: 0.0 },
  { A: 0.9, B: 0.2, C: 0.1 },
  { A: 0.2, B: 0.9, C: 0.8 },
  { A: 0.8, B: 0.6, C: 0.3 },
  { A: 0.7, B: 0.1, C: 0.9 },
  { A: 0.3, B: 1.0, C: 0.5 },
];

//================================================================
// 3. Підготовка даних та вивід за допомогою console.table()
//================================================================

// Створимо масив для зберігання результатів
const tableResults = [];

// Пройдемося по кожному набору даних і розрахуємо результат
testData.forEach((data, index) => {
  const { A, B, C } = data;

  // Проміжні обчислення
  const conjunction_BC = fuzzyAND(B, C);
  const negation_of_BC = fuzzyNOT(conjunction_BC);
  const finalResult = fuzzyIMP(A, negation_of_BC);

  // Додаємо новий об'єкт з результатами в наш масив
  tableResults.push({
    "№": index + 1,
    A: A,
    B: B,
    C: C,
    "B ∧ C": parseFloat(conjunction_BC.toFixed(2)),
    "¬(B ∧ C)": parseFloat(negation_of_BC.toFixed(2)),
    "f₉": parseFloat(finalResult.toFixed(2)),
  });
});

// Виводимо весь масив у вигляді гарної таблиці!
console.table(tableResults);
