# Протокол решающих контролей v1

Зафиксирован 27 октября 2026 для этапа S2. Статус: протокол готов; реализация,
проверка полных Inputs и launch review ещё предстоят. Новые расчёты не выполнены.
Протокол составлен после просмотра CIFAR-100N test outcomes: это exploratory
follow-up, а не preregistration невиданного теста. Новые изменения записываются
в следующую версию с причиной и датой; исходная версия сохраняется.

## 1. Вопросы

H1: early-mean teacher даёт student пользу сверх одного validation-selected
раннего snapshot при общей residual head и decision rule.
H2: дополнительная польза отличается от усреднения независимых моделей.
H3: residual student занимает полезное место на измеренной quality/cost кривой
относительно базы, teacher и двухпроходной интерполяции.

Early mean инвариантно к перестановке snapshots. Этот эксперимент не проверяет
порядок, форму полной траектории или необходимость всех её точек.
Выявленный confound: epochs1–5 принадлежат CE warmup, дальнейшее обучение — GCE.
Best base checkpoint может совпадать с одним из ранних snapshots. Записать его
эпоху для каждой базы. Причинное отделение времени от loss требует отдельного
fixed-loss контроля в S3/S4; S2 сам его не обеспечивает.

## 2. Общие условия

- CIFAR-100N/noisy100, 100 классов, исходные базы42/43/44. Новые базы не обучаем.
- Base split seed42; meta split seed20261015. Noisy holdout10% выбирает checkpoint
  базы и alpha. Clean official test используется для frozen evaluation.
- Recipient b всегда использует собственные frozen2048features и p_base.
- Meta fit60%, meta-val10%; зарезервированные15%/15% не добавляются в fit.
  Feature mean/scale вычисляются только на meta-fit rows; backbone/BN в eval.
- MLP2048→256→100, ReLU, centered residual, нулевой final layer.
  Posterior = softmax(log(p_base)+alpha*h(x)); alpha0 — точный identity.
- Head seed42, 25 epochs, AdamW lr0.001, wd0.0001, batch256, T1.
  Loss KL(q_teacher||p_student); checkpoint — min reserved meta-val KL,
  включая identity initialization epoch0.
- Alpha grid [0,0.25,0.5,1]: noisy-val accuracy, затем lower NLL, smaller alpha.
- Targets: canonical eval transforms и FP32; без TTA/train augmentation.
  Precision/software сохраняются. Clean train labels не участвуют нигде в fit,
  target construction, normalization или выборе параметров.

## 3. Ветки S2-A

| ID | Target / действие | Статус до S2 | Назначение |
|---|---|---|---|
| identity | Исходная база | Есть42/43/44 | Нулевая точка |
| early5 | Среднее probabilities epochs1–5 | Student есть42/43/44 | Основной кандидат без refit |
| selected1 | Один epoch1–5 по правилу ниже | Есть43/44;42 pending | Primary single-teacher contrast |
| early2 | Среднее probabilities epochs1и5, без выбора пары | Pending42/43/44 | Два teacher passes одной истории |
| independent2 | Среднее двух ДРУГИХ bases best checkpoints | Pending42/43/44 | Два независимых teacher passes, общая head |
| supervised_head | Та же residual head, noisy one-hot targets | Pending42/43/44 | Дополнительная head capacity и noisy supervision |
| direct_soft5 | Тот же early5 target, прямой softmax MLP без base-log offset | Pending42/43/44 | Обычная soft-target head против residual parametrization |
| self_identity | Target=p_base, zero residual | Structural check | Точное нулевое решение;25GPU epochs не нужны |

### Выбор selected1

Для каждого epoch1–5 оценить интерполяцию
softmax((1-alpha)log(p_base)+alpha log(q_epoch)) по общему alpha grid.
Ранжирование: noisy-val accuracy, lower NLL, smaller alpha, earlier epoch.
Обучить head на raw q выбранной эпохи; head alpha выбрать отдельно.
Это сохраняет правило завершённых controls43/44. Здесь20 teacher candidates
против4 у early5: teacher-selection search budget не одинаков, экспортировать grid.
Выбор teacher по его качеству не гарантирует оптимальный student — это ограничение
контроля. Fixed epoch1 результаты сохраняются, test-based best-of не строится.

### Независимые teachers

Donors: b42←43+44, b43←42+44, b44←42+43; recipient в его target не входит.
Проверить class order, decoded inputs, train/val membership, observed label vectors,
transforms. Переставленные IDs соединять явно verified join по IDs.
При отсутствии cached donor probabilities извлечь их canonical FP32 из проверенных
весов; не менять split. Target — arithmetic probability mean с row normalization.

Общий ImageNet initialization recipe не равен независимым pretraining runs.
Donor overlap делает три independent2 recipient comparisons зависимыми;
не выдавать их за независимую выборку ансамблей для inferential claims.
Early2 и independent2 сопоставимы по числу teacher evaluations/head budget,
но независимые bases требуют дополнительных training runs. Early5 против двух
моделей также не equal-compute. Публиковать обе оси стоимости.

### Head-only controls

Supervised_head: CE по observed noisy labels на тех же meta-fit rows; min noisy
meta-val CE checkpoint; те же optimizer/epochs/alpha. Это иная supervision,
поэтому контроль capacity не является exact-KL objective control.
Train labels восстановить по IDs из проверенного noisy dataset, не из clean labels.

Direct_soft5: features, width, teacher, KL, optimizer, epochs и мета-split те же.
Прямой softmax h(x), без log(p_base) offset; zero final layer означает uniform
начальное распределение. При evaluation alpha интерполирует log p_base и log q_head
тем же правилом. Initialization отличается структурно: это часть сравнения
parametrization, её нельзя объявлять строго одинаковой инициализацией posterior.
Не подбирать специально более выгодный lr/epochs для одного варианта.

Для всех teacher targets сохранить raw teacher и validation-selected teacher/base
интерполяцию. Это дополнительные reference rows, не новые обучаемые кандидаты.

## 4. Порядок и ресурсы

1. CPU preflight полных Outputs всех трёх recipient bases: train/val evidence,
   checkpoints, snapshots1–5, source configs/policies, original raw predictions,
   CIFAR input. Summary-only архив недостаточен.
2. До любого fit проверить IDs/splits/classes/digests. При пропуске записать
   missing_input и точный список файлов, не подменять базу или seed.
3. Оценить extraction/head ресурсы коротким train/val smoke; smoke не считается
   complete result. Проверить квоту один раз. Фиксированный бюджет не урезать
   после просмотра метрик; недостающий бюджет — deferred, а не успех.
4. Existing early5 и selected1(43/44) использовать без refit.
5. ПакетA: selected1(42), early2, supervised_head, direct_soft5. ПакетB:
   independent2. Пакеты можно выполнять в двух ноутбуках, если Inputs/квота позволяют.
6. Каждый пакет замораживает все свои policies до первого своего test evaluation.
   Пакеты не используют test outcomes друг друга для изменения настроек.
7. Sources read-only; новые target caches отдельно. Bounded batches, mmap/streamed
   probability sums; не stack всех full image tensors в RAM.
8. Один final test на frozen policy; сохранить все исходы. Сбой может возобновить
   расчёт с теми же policies. Статус complete только при проверяемых outputs.
9. Подтвердить submission один раз и прекратить UI/log polling. Монитор не создавать.

Новых fit максимум13: selected1(42) +4семейства×3базы. GPU-часы до smoke не
назначены; многократное извлечение image features может стоить больше head fitting.

## 5. S2-B: реальная стоимость

Методы на одном устройстве/software/FP32: base, residual student,
selected1+base, early2, early5, independent2. Student извлекает features/logits
ОДНИМ backbone forward. Base-logits и residual-head должны делить те же features.

- Batch sizes1и32;32 фиксированных noisy-val images, без test tuning.
- Model-only: inputs уже на GPU. Pipeline: decoded CPU images→transform→transfer
  →output tensor. Network download/распаковка и cold loading сюда не входят.
- Warmup20 batches;10 measurement blocks×50 batches; CUDA synchronization до/после
  timed scope. Порядок методов перемешать с seed20261015 для каждого блока.
- Eval/inference_mode; output consumption одинаковое. Сохранить block medians,
  p10/p90, ms/image, ms/batch, images/s, peak allocated/reserved GPU memory.
- Blocks не являются независимыми hardware sessions. OOM записать явно; не
  уменьшать batch только для одного метода в общей таблице.
- Основной teacher throughput: weights resident. Совпадающие base/snapshot digests
  использовать одним forward, а не искусственно считать двумя сетями. Записать
  nominal и фактическое число уникальных backbone passes. Cold loading/host memory/filebytes
  отдельно. Не смешивать startup и steady-state.

Training cost: base epoch time, snapshot write/storage, canonical teacher/feature
extraction, head fit, policy selection. Неизмеренное = unavailable, не0.
Wall time и GPU-hours учитывать по фактически использованным устройствам.
CUDA witness на двух GPU не доказывает параллельного ускорения workload.

## 6. Анализ и решение

Primary: accuracy early5student − selected1student, отдельно для каждой базы,
равновесное среднее по3базам и sample SD. Это exploratory contrast известного теста.
Secondary: early2−independent2, early5−supervised_head, early5−direct_soft5,
teacher/student gap, NLL/Brier/ECE15bins, AURC, own-error AUROC и fixed-base-error
AUROC. Events не смешивать. Для каждой correction: corrected/harmed/unchanged,
changedtotal и alpha0 outcomes.

Conditional paired-example intervals:2000 bootstrap, seed20261015, nominal95%.
Не объединять три оценки одного10k test в30k независимых observations.
Межбазовая неопределённость при n3 ограничена. Нет confirmatory p-values или
выбора победителя по набору secondary metrics. Для S4 задаётся отдельный фиксированный
размер репликации до подтверждающих результатов; не добавлять seeds до significance.

S2 — описательное решение при n3, без автоматического численного PASS/FAIL.
Минимальный практически значимый эффект и размер подтверждающего исследования
фиксируются отдельно до S4; текущие известные эффекты не объявляются confirmation.
Если single/static controls объясняют выигрыш, снять temporal-specific accuracy
претензию. Если quality близка multi-pass и стоимость ниже, исследовать compression
trade-off; это само по себе не доказывает новизну. Calibration degradation и
negative transfer сохраняются. Новая архитектура требует записанной гипотезы S3.

Близкий SelfPLL baseline закреплён для S3 implementation review официального кода
и adaptation к нашему feature/noisy-val протоколу. Supervised_head/identity не
являются репликацией SelfPLL. Для S4 зарегистрирован robust-training reference
DivideMix; актуальную дополнительную альтернативу выбрать до confirmation после
проверки code/data/pretraining budget. GCE-alone не подтверждает современное SOTA.

## 7. Outputs

Resolved config.yaml, metrics.json, run_metadata.json, report.md; head checkpoints,
policies/selection grids, source/target manifests, raw predictions, compact tables,
runtime samples и resource status. Каждая ветка имеет pending/running/complete/
missing_input/failed/deferred. Все selections, включая отрицательные, экспортируются.

Это scientific protocol freeze. Implementation tests и независимый launch review
обязательны до расчётов; они могут выявить необходимость явной версии v2.

Проверка совместимости27октября: исходные ZIP пилота, обеих replications и
selected-control содержат epoch0 initialization в checkpoint competition.
Результат проверки: `outputs/conference_stage1_20261027/protocol_checks.json`.
