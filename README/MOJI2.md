# ParkinsonAiResearch --- راهنمای تحویل و شروع فاز Preprocessing

> **B1 update:** Preprocessing and data loading are now implemented. The current
> usage, verification results and limitations are in [B1_PREPROCESSING.md](B1_PREPROCESSING.md).
> The active implementation is in `data/dataloader.py`, with settings in `config/data_config.py`
> and review helpers in `utils/`, following the supplied project layout. Descriptions below of a
> separate preprocessing module refer to the earlier A-phase proposal.


## 1. وضعیت فعلی پروژه

این مخزن زیرساخت داده و baseline اولیه برای پژوهش تشخیص/غربالگری
پارکینسون از روی ترسیم مارپیچ است.

در نسخه فعلی، بخش‌های زیر در کد وجود دارند:

-   بررسی آرشیوهای ZIP و ساخت Inventory برای شرکت‌کنندگان، تصاویر PNG و
    فایل‌های TXT؛
-   ساخت جدول Labelهای متخصصان و ثبت سوابق رأی‌ها؛
-   ساخت پروتکل Fold از روی `folding_TLO.xlsx`؛
-   تولید `fold_manifest.csv` برای اتصال تصویر، Subject، Label و نقش هر
    نمونه در هر Fold؛
-   کنترل نشت اطلاعات در سطح Subject و محتوای تکراری؛
-   پیاده‌سازی `SpiralDataset` و `DataLoader` در PyTorch؛
-   baseline تصویری ResNet و اسکریپت‌های آموزش و ارزیابی؛
-   اعتبارسنجی خروجی‌های Annotation و Fold.

**Preprocessing پژوهشی هنوز پیاده‌سازی نشده است.** پوشه `preprocessing/`
در حال حاضر فقط فایل `__init__.py` دارد. تبدیل موجود در Dataset یک تبدیل
پایه برای ورود تصویر به CNN است و نباید به‌عنوان روش نهایی پژوهش گزارش
شود.



مسئولیت پوشه‌ها:

  مسیر                      مسئولیت
  ------------------------- -------------------------------------------------
  `datasets/raw/`           فایل‌های خام استخراج‌شده؛ نباید تغییر داده شوند
  `datasets/annotations/`   CSVها، Labelها، Foldها و گزارش‌های Audit
  `data/`                   خواندن داده، Label، Manifest، Fold و Dataset
  `preprocessing/`          محل پیاده‌سازی تبدیل‌های تصویری/سری‌زمانی
  `model/`                  معماری مدل
  `utils/`                  حلقه آموزش، معیارها، ذخیره فایل و بازتولیدپذیری
  `outputs/`                Checkpointها و خروجی‌های آموزش/ارزیابی

پوشه‌های `datasets/` و `outputs/` در `.gitignore` هستند؛ بنابراین فایل‌های
واقعی داده و خروجی مدل نباید به‌عنوان بخشی از کد منبع به Git اضافه شوند.

------------------------------------------------------------------------

## 4. جریان ساخت داده و فایل‌های خروجی

سه مرحله مستقل برای آماده‌سازی داده وجود دارد.

### 4.1 بررسی آرشیوها

ورودی‌ها: - ZIP گروه Control - ZIP گروه Parkinson

اسکریپت:

``` bash
python audit_data.py \
  --control-zip /path/to/control.zip \
  --parkinson-zip /path/to/parkinson.zip
```

خروجی‌های اصلی در `datasets/annotations/`:

-   `participants.csv`
-   `drawings.csv`
-   `trace_inventory.csv`
-   `duplicate_ledger.csv`
-   `trace_duplicate_ledger.csv`
-   `data_audit_report.json`

تصاویر و TXTها، در حالت پیش‌فرض، زیر `datasets/raw/` استخراج می‌شوند. اگر
داده‌ها از قبل در مسیر مورد انتظار استخراج شده‌اند، گزینه `--no-extract`
قابل استفاده است؛ در این حالت فقط ZIP بررسی می‌شود و فایل‌های روی دیسک
ساخته نمی‌شوند.

### 4.2 ساخت Label متخصصان

ورودی‌ها: - CSV کامل Control - CSV کامل Parkinson

``` bash
python finalize_labels.py \
  --control-csv /path/to/complete_control.csv \
  --parkinson-csv /path/to/complete_parkinson.csv
```

خروجی‌ها:

-   `annotations_long.csv`
-   `labels.csv`
-   `doctor_registry.csv`
-   `label_audit_report.json`

قانون فعلی Consensus:

1.  اگر هر سه پزشک باقی‌مانده Parkinson رأی داده باشند، Label برابر
    `parkinson` است.
2.  اگر حداقل دو پزشک از سه پزشک Control رأی داده باشند، Label برابر
    `control` است.
3.  در سایر الگوهای رأی معتبر، Label برابر `unknown` است.

پزشک D1 از Consensus کنار گذاشته شده، اما رأی خام او در سوابق منبع حفظ
می‌شود. پزشکان D2، D3 و D4 در Consensus شرکت دارند. Confidence به‌عنوان
Metadata حفظ می‌شود و در نسخه فعلی آستانه‌ای برای حذف یا وزن‌دهی رأی‌ها
اعمال نمی‌شود. رأی تکراری یک پزشک برای یک تصویر، در صورت یکسان بودن
Label، فقط یک رأی محسوب می‌شود و سوابق تکرار حذف نمی‌شوند.

دو نوع Label را با هم اشتباه نگیرید:

-   `expert_label`: نظر اجماعی متخصصان درباره تصویر؛ سه‌کلاسه شامل
    `control`، `parkinson` و `unknown`.
-   `clinical_label_recorded`: گروه بالینی/پوشه ثبت‌شده برای فرد؛ دودویی
    و مبتنی بر گروه `control` یا `parkinson`.

### 4.3 ساخت Foldها و Manifest نهایی

ورودی‌ها: - `participants.csv` - `drawings.csv` - `labels.csv` - فایل
`folding_TLO.xlsx`

``` bash
python build_folds.py \
  --workbook /path/to/folding_TLO.xlsx
```

خروجی‌ها:

-   `fold_assignments.csv`
-   `splits.csv`
-   `fold_manifest.csv`
-   `fold_summary.csv`
-   `fold_build_report.json`
-   `decision_register.csv`

`fold_manifest.csv` مرجع اصلی Dataset در نسخه فعلی است. فایل قدیمی
`manifest.csv` و Split تصادفی، مرجع اجرای فعلی نیستند.

**نکته:** اسکریپت `build_folds.py` قبل از ساخت خروجی، وجود Workbook و
CSVهای ورودی را بررسی می‌کند. فایل `labels.csv` باید برای تمام Drawingهای
موجود در `drawings.csv` رکورد متناظر داشته باشد؛ Label ناقص یا Drawing
بدون Label نباید با حدس تکمیل شود.

------------------------------------------------------------------------

## 5. قرارداد Fold و مواردی که نباید تغییر کنند

طبق منطق فعلی `data/fold_protocol.py`:

-   Workbook Foldهای 1 تا 25 را برای توسعه/اعتبارسنجی استفاده می‌کند.
-   Workbook Foldهای 26 تا 34 مجموعه Test قفل‌شده را تشکیل می‌دهند.
-   50 شرکت‌کننده در بخش توسعه و 18 شرکت‌کننده در فهرست Test قرار دارند.
-   در خروجی Roster، دو شرکت‌کننده Unassigned و یک شرکت‌کننده خارج از
    Workbook نیز ثبت می‌شوند.
-   مجموعه Test در تمام 25 Fold توسعه با نقش `test_locked` ثابت می‌ماند و
    نباید در انتخاب مدل یا تنظیم پارامترهای Preprocessing استفاده شود.

موارد خاص ثبت‌شده در کد:

-   **Fold 22:** دو شرکت‌کننده اعتبارسنجی آن در Export فعلی تصویر واجد
    شرایط ندارند. Fold باید در پروتکل حفظ شود؛ تصویر جایگزین، پیش‌بینی یا
    امتیاز Validation مصنوعی تولید نکنید.
-   **Control Subject 09:** در Workbook و فهرست Test قرار دارد، اما در
    آرشیو ارائه‌شده تصویر ندارد. آن را با فرد دیگری جایگزین نکنید.
-   **Control Subject 36:** به‌دلیل مسئله هویت/تصاویر تکراری با Control
    Subject 34، خارج از Foldها و در وضعیت Unassigned نگه داشته شده است.
-   **Parkinson Subject 31:** به‌دلیل نیاز به اصلاح/تأیید تشخیص بالینی،
    Unassigned باقی مانده است.
-   **Parkinson Subject 11:** در آرشیو وجود دارد، اما در Workbook نیست و
    تصویر مارپیچ ندارد؛ وارد Foldها نمی‌شود.
-   Duplicateهای دقیق درون یک Subject در وضعیت Quarantine قرار می‌گیرند و
    `cv_eligible=false` دارند.

هیچ‌کدام از موارد بالا را با حذف ردیف، تغییر شناسه، جابه‌جایی Subject یا
ساخت نمونه مصنوعی حل نکنید. هر تغییر در این تصمیم‌ها باید با تأیید مسئول
پژوهش و نسخه‌بندی جدید داده انجام شود.

------------------------------------------------------------------------

## 6. ساختار `fold_manifest.csv`

هر Drawing برای هر یک از 25 Fold توسعه یک ردیف دارد. بنابراین یک
`drawing_id` در کل Manifest می‌تواند 25 بار تکرار شود؛ این تکرار خطا
نیست.

برای شناسایی یک ردیف در Manifest از ترکیب زیر استفاده کنید:

``` text
(development_model_fold, drawing_id)
```

ستون‌های مهم:

  ستون                        مفهوم
  --------------------------- -----------------------------------------------------
  `development_model_fold`    شماره Fold توسعه، از 1 تا 25
  `drawing_id`                شناسه یکتای تصویر
  `participant_id`            شناسه یکتای فرد، همراه با Cohort
  `cohort`                    `control` یا `parkinson`
  `image_name`                نام فایل PNG
  `image_path`                مسیر PNG خام
  `partition`                 بخش اصلی Workbook مانند development یا heldout_test
  `workbook_fold_id`          Fold اصلی فرد در Workbook
  `role`                      نقش نمونه در Fold جاری
  `cv_eligible`               مجاز بودن نمونه برای CV
  `duplicate_group`           شناسه گروه تصاویر تکراری، در صورت وجود
  `image_sha256`              Hash محتوای PNG
  `expert_label`              Label اجماعی
  `label_status`              وضعیت نهایی‌سازی Label
  `clinical_label_recorded`   گروه بالینی ثبت‌شده

مقادیر مهم `role`:

-   `train`
-   `validation`
-   `test_locked`
-   `excluded_unassigned`
-   `excluded_outside_workbook`
-   `quarantined`

`SpiralDataset` فقط نمونه‌های دارای نقش مورد درخواست و `cv_eligible=true`
را انتخاب می‌کند.

------------------------------------------------------------------------

## 7. قرارداد فعلی Dataset و اتصال Preprocessing

کلاس اصلی در `data/dataset.py` قرار دارد:

``` python
from data.dataset import SpiralDataset

dataset = SpiralDataset(
    manifest_path=DATA.fold_manifest_path,
    root=ROOT_DIR,
    fold_id=1,
    split="train",
    label_source="consensus",
)
```

مقادیر مجاز:

-   `fold_id`: عدد صحیح بین 1 تا 25
-   `split`: یکی از `train`، `val` یا `test`
-   `label_source`: یکی از `consensus` یا `real`

تطبیق Split با Manifest:

  ورودی Dataset   مقدار `role` در Manifest
  --------------- --------------------------
  `train`         `train`
  `val`           `validation`
  `test`          `test_locked`

هر نمونه خروجی Dataset به شکل Dictionary است و کلیدهای اصلی آن عبارت‌اند
از:

-   `image`: Tensor تصویر
-   `label`: شناسه عددی کلاس
-   `label_name`: نام کلاس
-   `drawing_id`
-   `image_name`
-   `participant_id`
-   `subject_key`
-   `fold_id`
-   `role`

### رفتار مهم پارامتر `transform`

پارامتر `transform` در `SpiralDataset` کل تبدیل پیش‌فرض را **جایگزین**
می‌کند؛ به‌صورت خودکار بعد از Transform سفارشی، `BaselineImageTransform`
اجرا نمی‌شود.

Transform سفارشی باید:

1.  یک تصویر `PIL.Image` دریافت کند؛
2.  خروجی Tensor عددی با ترتیب ابعاد `C × H × W` برگرداند؛
3.  برای تمام نمونه‌ها اندازه و تعداد کانال سازگار تولید کند؛
4.  مقادیر خروجی محدود و finite باشند؛
5.  در صورت انجام Normalization، دقیقاً مشخص کند از چه پارامترهایی
    استفاده شده است.

تبدیل پایه فعلی (`BaselineImageTransform`) این کارها را انجام می‌دهد:

1.  تبدیل به RGBA؛
2.  ترکیب Alpha با پس‌زمینه سفید و تبدیل به RGB؛
3.  قرار دادن تصویر در قاب مربعی 224×224 با حفظ نسبت تصویر و Padding؛
4.  تبدیل به Tensor؛
5.  Normalization با Mean و Standard Deviation استاندارد ImageNet.

این تبدیل فقط Baseline است؛ روش علمی Preprocessing پروژه محسوب نمی‌شود.

### محل اتصال در Train و Eval

در نسخه فعلی، `train.py` و `eval.py` هنگام فراخوانی `make_loader`،
Transform سفارشی دریافت نمی‌کنند. بنابراین حتی اگر ماژول Preprocessing
نوشته شود، تا زمانی که این دو اسکریپت به آن متصل نشوند، آموزش و ارزیابی
همچنان از تبدیل پایه Dataset استفاده خواهند کرد.

الگوی اتصال مورد انتظار:

``` python
# الگوی مفهومی؛ نام توابع نهایی پس از پیاده‌سازی Preprocessing تثبیت شود.
train_transform = build_train_transform(...)
eval_transform = build_eval_transform(...)

train_loader = make_loader(
    fold_id=fold_id,
    split="train",
    transform=train_transform,
    ...
)

val_loader = make_loader(
    fold_id=fold_id,
    split="val",
    transform=eval_transform,
    ...
)
```

در `eval.py` نیز باید از همان `build_eval_transform` استفاده شود.
پیاده‌سازی نهایی باید با مسئول Dataset/Training هماهنگ و با تست پوشش داده
شود.

------------------------------------------------------------------------

## 8. وظایف فاز Preprocessing

هدف این فاز طراحی و ارزیابی علمی تبدیل‌های مناسب است؛ نه اینکه تمام
روش‌های رایج بدون بررسی روی تصاویر اعمال شوند.

موارد قابل بررسی:

-   بررسی Background و Foreground؛
-   Crop ناحیه مارپیچ و حذف حاشیه‌های غیرضروری؛
-   Centering و Deskewing؛
-   Denoising؛
-   Thresholding و Binarization؛
-   Contrast Enhancement؛
-   عملیات Morphological؛
-   Normalization؛
-   Augmentation؛
-   در صورت تأیید Pairing سری زمانی، بازسازی یا تولید نمایش تصویری مبتنی
    بر زمان.

### اصول ضروری

1.  فایل‌های خام PNG و TXT را هرگز Overwrite نکنید.
2.  تبدیل‌های قطعی و یکسان را برای Train، Validation و Test اعمال کنید.
3.  Augmentation تصادفی فقط برای Train مجاز است؛ Validation و Test باید
    deterministic باشند.
4.  اگر پارامتر یا آماره‌ای از کل داده تخمین زده می‌شود، این کار می‌تواند
    Leakage ایجاد کند. هر پارامتر آموختنی باید فقط از داده Train همان
    Fold محاسبه شود.
5.  Test قفل‌شده را برای انتخاب نوع Preprocessing، آستانه‌ها یا
    Hyperparameterها بررسی نکنید.
6.  Flip، Rotation، تغییر جهت Stroke یا هر تبدیل تغییردهنده ویژگی حرکتی
    را بدون توجیه علمی و تأیید پژوهشی اعمال نکنید.
7.  خروجی هر مرحله را روی نمونه‌های مختلف Control و Parkinson به‌صورت
    تصویری بررسی و نسخه روش را ثبت کنید.
8.  پارامترها، ترتیب مراحل، اندازه خروجی، تعداد کانال‌ها، روش
    Normalization و Seed مربوط به Augmentation باید مستند باشند.

### پیشنهاد ساختار ماژول

این ساختار صرفاً پیشنهاد برای توسعه است و هنوز در مخزن پیاده‌سازی نشده:

``` text
preprocessing/
├── __init__.py
├── pipeline.py
├── transforms.py
├── visualization.py
└── tests/
    └── test_preprocessing.py
```

بهتر است دو مسیر مستقل تعریف شود:

-   `build_train_transform(...)`: تبدیل‌های قطعی به‌علاوه Augmentation
    مخصوص Train؛
-   `build_eval_transform(...)`: فقط تبدیل‌های قطعی و deterministic برای
    Validation/Test.

------------------------------------------------------------------------

## 9. نکته مهم درباره فایل‌های TXT و داده زمانی

در حال حاضر:

-   `trace_inventory.csv` مسیر و Hash فایل‌های TXT را نگه می‌دارد.
-   در `drawings.csv` مقدار `trace_path` برای تصاویر خالی است.
-   وضعیت Pairing برابر `unverified_no_explicit_drawing_id` ثبت شده است.
-   برابر بودن تعداد PNG و TXT یا مرتب‌سازی نام فایل‌ها، اثبات نمی‌کند هر
    دو متعلق به یک Drawing هستند.
-   Transform فعلی فقط تصویر PIL را دریافت می‌کند و اطلاعات `drawing_id`
    یا مسیر TXT را به‌عنوان ورودی نمی‌گیرد.

بنابراین تا زمانی که نگاشت معتبر و قابل بازتولید PNG به TXT ساخته و
تأیید نشده است، از TXT برای ساخت Time-Augmented Spiral یا هر Feature
زمانی استفاده نکنید.

اگر قرار است TXT وارد مدل شود، لازم است ابتدا یک جدول Pairing مستقل و
قابل ممیزی، با کلید `drawing_id` و شناسه فایل TXT طراحی شود؛ سپس Dataset
به‌صورت صریح برای خواندن هر دو Modality توسعه یابد. هیچ Pairing حدسی بر
اساس تعداد یا ترتیب فایل‌ها مجاز نیست.

------------------------------------------------------------------------

## 10. اعتبارسنجی و تست

تمام فرمان‌ها را از ریشه Repository اجرا کنید تا مسیرهای نسبی با تنظیمات
پروژه سازگار باشند.

### تست‌های خودکار

``` bash
pytest -q
```

تست‌های موجود در مخزن، منطق Label و Fold/Workbook و یک مورد
Duplicate-content leakage را پوشش می‌دهند. در نسخه فعلی، تست اختصاصی برای
`SpiralDataset`، Transformهای Preprocessing و اتصال Train/Eval وجود
ندارد؛ این تست‌ها باید در همین فاز اضافه شوند.

### اعتبارسنجی فایل‌های واقعی

``` bash
python validate_data.py
```

این فرمان وجود فایل‌های تصویری اشاره‌شده در CSVها را نیز بررسی می‌کند. اگر
فقط می‌خواهید اتصال CSVها و Manifest را بررسی کنید و تصاویر هنوز روی دیسک
نیستند:

``` bash
python validate_data.py --skip-image-files
```

### اجرای Smoke Test آموزش

پس از اتصال Preprocessing به Train:

``` bash
python train.py --fold-id 1 --epochs 1 --batch-size 2 --num-workers 0
```

برای Fold 22 انتظار Validation metric نداشته باشید؛ این Fold در پروتکل
حفظ شده اما Validation image ندارد.

ارزیابی Test فقط پس از تثبیت روش و تصمیم نهایی پژوهشی انجام شود. Test
نباید در طول توسعه برای انتخاب Preprocessing یا مدل استفاده شود.

------------------------------------------------------------------------

## 11. تست‌هایی که مسئول Preprocessing باید اضافه کند

حداقل موارد زیر باید تست شوند:

-   خروجی Transform یک Tensor با شکل ثابت `C × H × W` باشد؛
-   خروجی برای ورودی‌های PNG با Modeهای RGB و RGBA معتبر باشد؛
-   تصویر خراب یا ورودی نامعتبر به خطای قابل فهم منجر شود؛
-   Transform ارزیابی برای ورودی یکسان خروجی یکسان تولید کند؛
-   Augmentation فقط در مسیر Train فعال باشد؛
-   فایل خام بعد از اجرای Pipeline تغییر نکند؛
-   خروجی‌ها NaN یا Inf نداشته باشند؛
-   مسیر Train و Validation از یک Manifest و Fold یکسان، اما Transform
    مناسب همان Split استفاده کنند؛
-   هیچ Subject یا Duplicate group واجد شرایطی بین Train و
    Validation/Test نشت نکند؛
-   اگر Cache آفلاین ساخته می‌شود، ارتباط هر خروجی با `drawing_id` و نسخه
    Pipeline قابل ردیابی باشد.

------------------------------------------------------------------------

## 12. قواعد همکاری و تحویل

هنگام توسعه Preprocessing:

-   منطق `data/fold_protocol.py` و تخصیص Foldها را بدون هماهنگی تغییر
    ندهید.
-   ستون‌های هویتی و Label در `fold_manifest.csv` را بازنویسی نکنید.
-   فایل‌های خام را حذف یا جایگزین نکنید.
-   تغییر در Label policy، Split policy یا حذف نمونه‌ها باید جداگانه
    تأیید و نسخه‌بندی شود.
-   اگر فایل‌های پردازش‌شده ذخیره می‌شوند، آن‌ها را در مسیر جداگانه مانند
    `datasets/processed/<pipeline_version>/` قرار دهید؛ `image_path` خام
    را از بین نبرید.
-   برای هر Pipeline، نام نسخه، ترتیب مراحل، پارامترها و Hash/شناسه
    ورودی و خروجی را ثبت کنید.
-   پس از تکمیل، کد Preprocessing، تست‌ها، نمونه تصاویر قبل/بعد و توضیح
    تصمیم‌های علمی را تحویل دهید.

### معیار پایان فاز

فاز Preprocessing زمانی آماده اتصال به مدل است که:

1.  روش و دلیل انتخاب هر مرحله مستند باشد؛
2.  خروجی روی نمونه‌های متنوع به‌صورت بصری بررسی شده باشد؛
3.  Transformهای Train و Eval از هم جدا و deterministic بودن Eval تأیید
    شده باشد؛
4.  تست‌های واحد Preprocessing موفق باشند؛
5.  `train.py` و `eval.py` واقعاً از Transformهای جدید استفاده کنند؛
6.  Manifest، Labelها و Foldها بدون تغییر ناخواسته باقی مانده باشند؛
7.  `pytest -q` و `python validate_data.py` روی محیط داده واقعی اجرا و
    خروجی آن ثبت شده باشد.

------------------------------------------------------------------------

## 13. مرجع فایل‌های اصلی

  موضوع                      فایل
  -------------------------- -------------------------
  مسیرها و تنظیمات           `config/settings.py`
  Audit آرشیو و تصاویر       `data/audit.py`
  Label متخصصان              `data/labels.py`
  Workbook و ساخت Fold       `data/fold_protocol.py`
  Dataset و Transform پایه   `data/dataset.py`
  کنترل Leakage              `data/splits.py`
  اجرای Audit                `audit_data.py`
  ساخت Labelها               `finalize_labels.py`
  ساخت Manifest              `build_folds.py`
  اعتبارسنجی                 `validate_data.py`
  آموزش                      `train.py`
  ارزیابی                    `eval.py`
  محل توسعه Preprocessing    `preprocessing/`

**اصل نهایی:** Preprocessing باید روی داده خام و با حفظ کامل هویت
Drawing، Subject، Label و Fold انجام شود. هیچ تبدیل تصویری نباید مرز
Train/Validation/Test یا قابلیت ردیابی داده را مخدوش کند.
