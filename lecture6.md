class: middle, center, title-slide

# Методи чисельної оптимізації

Лекція 6: Адаптивні методи оптимізації
<br><br>
Кочура Юрій Петрович<br>
[iuriy.kochura@gmail.com](mailto:iuriy.kochura@gmail.com) <br>
<a href="https://t.me/y_kochura">@y_kochura</a> <br>

---

class:  black-slide, 
background-image: url(./figures/lec1/blog-header-cost-optimization-examples.jpg)
background-size: cover

# Сьогодні

.larger-x[ <p class="shadow" style="line-height: 200%;">Як ефективно оптимізувати параметри? <br>
  Адаптивні методи: <br>

🎙️ AdaGrad <br>
🎙️ RMSProp  <br>
🎙️ Adam <br> 
</p>]

---


class: blue-slide, middle, center
count: false

.larger-xx[Адаптивні методи]

---

class: middle

# Адаптивні методи  

Величина градієнтів часто сильно відрізняється між шарами нейронної мережі, тому глобальна швидкість навчання може не працювати належним чином.

*Загальна ідея.* Замість того, щоб використовувати однакову швидкість навчання для кожної ваги в нашій мережі, **підтримуйте оцінку кращої швидкості окремо для кожної ваги**.

Точний спосіб адаптації до швидкості навчання залежить від алгоритму, але більшість методів або **пристосовуються** до **дисперсії ваг**, або до **локальної кривизни** проблеми.

.footnote[Credits: Aaron Defazio, Facebook AI Research]

---


class: middle

## AdaGrad

Зменшення масштабу для кожного параметра на квадратний корінь із суми квадратів усіх його історичних значень.

$$\begin{aligned}
r\_t  &=  r\_{t-1} + g\_t \odot g\_t \\\\
W\_{t+1} &= W\_t - \frac{\alpha}{\varepsilon + \sqrt{r\_t}} \odot g\_t
\end{aligned}$$

- AdaGrad позбавляє від необхідності вручну налаштовувати швидкість навчання.
  Більшість реалізацій використовують $\alpha=0.01$ за замовчуванням.
- Добре, коли цільова функція опукла.
- $r_t$ необмежено зростає під час навчання, що може спричинити зменшення розміру кроку та зрештою стати нескінченно малим.
- $\varepsilon$ адитивна константа, яка гарантує, що ми не ділимо на 0.


???
The variable $r\_t$ accumulates past gradient variance.

---


class: middle

## RMSProp

Те саме, що AdaGrad, але накопичує експоненціально спадне середнє значення градієнта.

**Ключова ідея.** Нормалізувати за середньоквадратичним значенням градієнта

$$\begin{aligned}
r\_t  &=  \rho r\_{t-1} + (1-\rho) g\_t \odot g\_t \\\\
W\_{t+1} &= W\_t - \frac{\alpha}{\varepsilon + \sqrt{r\_t}} \odot g\_t
\end{aligned}$$

- Ефективніший, коли цільова функція не є опуклою.

???
An alternative is to use a leaky average in the same way we used in the momentum method, for some parameter $\rho>0$.

---


class: middle

## Adam: RMSprop з імпульсом

.smaller-xx[“Adaptive Moment Estimation”]

Подібно до RMSProp з імпульсом, але з умовами корекції зсуву для першого та другого моментів.

$$\begin{aligned}
p\_t  &=  \rho\_1 p\_{t-1} + (1-\rho\_1) g\_t \\\\
\hat{p}\_t &= \frac{p\_t}{1-\rho\_1^t} \\\\
r\_t  &=  \rho\_2 r\_{t-1} + (1-\rho\_2) g\_t \odot g\_t \\\\
\hat{r}\_t &= \frac{r\_t}{1-\rho\_2^t} \\\\
W\_{t+1} &= W\_t - \alpha \frac{\hat{p}\_t}{\varepsilon+\sqrt{\hat{r}\_t}}
\end{aligned}$$

- Хороші значення за замовчуванням $\rho\_1=0.9$ і $\rho\_2=0.999$.
- Подібно до того, як імпульс покращує SGD, він також покращує RMSProp.

???
Adam is one of the **default optimizers** in deep learning, along with SGD with momentum.

---

# Практична сторона

Для погано обумовлених задач Adam часто набагато кращий, ніж SGD.

Використовуйте Adam замість RMSprop завдяки очевидним перевагам імпульсу.

## Але, Adam погано вивчений теоретично та має відомі недоліки:

- Зовсім не сходиться на деяких простих прикладах задач!
- Дає гіршу помилку узагальнення для багатьох задач комп’ютерного зору (наприклад, ImageNet)
- Потрібно більше пам'яті, ніж SGD
- Має 2 параметри моментів, тому може знадобитися додаткове налаштування

.footnote[Credits: Aaron Defazio, Facebook AI Research]

---

class: middle, center

.larger-xx[[Демо](https://www.deeplearning.ai/ai-notes/optimization/)]

---

class: middle, center

.larger-xx[[demo - losslandscape](https://losslandscape.com/explorer)]

---

class: end-slide, center
count: false

.larger-xxxx[🏁]


