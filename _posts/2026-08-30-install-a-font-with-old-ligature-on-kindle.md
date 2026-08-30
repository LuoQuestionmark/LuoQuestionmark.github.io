---
layout: post
title: "Install a Font with Old Ligature on Kindle"
date: 2026-08-30 14:30:00 +0800
categories: Typography
---

<style>
  @font-face {
    font-family: "KpRoman-Oldstyle";
    src: url("{{ 'assets/fonts/KpRoman-Regular-Oldstyle.ttf' | relative_url}}") format("truetype");
  }

  p {
    font-family: "KpRoman-Oldstyle", Courier, monospace;
  }
</style>

## Inspiration & Motivation

A few years ago, I bought some high-end books in France. These books are parts of the *Bibliothèque de la Pléiade*, with very thin papers, a leather cover, and very curious typesets: extra ligatures for the combinations of letters such as "ct" "st". For some OpenType fonts, these are considered as "[discretionary ligature](https://en.wikipedia.org/wiki/Ligature_(writing)#Computer_typesetting)". As the name suggests, less common compare to more popular ligatures, such as ff, fi, ffi.

I wanted to install such a font on my Kindle, and this article explain how I did it. Actually, the particular article use the exact font that I have configured based on the font [KpRoman](https://ctan.org/tex-archive/fonts/kpfonts-otf). (It took me two hours to make it work on Jekyll, I hate Web programming.)

## Selection of Fonts

The first font I found with such ligatures is [Adobe Minion](https://fonts.adobe.com/fonts/minion) family. However, this is a paid font. Following suggestion from LLM, I found the KpRoman fonts, which also looks very nice and can be downloaded from [CTAN](https://ctan.org/tex-archive/fonts/kpfonts-otf). The creator kindly put these fonts under SIL Open Font License, which means it can be used freely on my Kindle.

Here is how it looks like on my Kindle:
![Kindle Screen Capture with KpRoman configured](/../../../../../assets/images/kindle_screenshot.png)

## Install Fonts on Kindle

To install a font on Kindle is quite easy: just copy-paste the font file (support `ttf` or `otf` format) to the `fonts` folder on Kindle. The rest is to find a font that grants relavant ligatures. As mentioned earlier, for OpenType fonts, these ligatures are considered as "discretionary ligature" and can be activated with `dlig` option enabled.

However, while it is relatively easy to enable such feature with CSS file or TeX, Kindle readers do not provide free access to toggle `dlig` features, which are disabled by default. Although embedding `.css` file while exporting ebook (with supported format) from calibre to Kindle works on calibre ebook reader, but not on Kindle.

My approach is to modify the KpRoman fonts so that these ligatures are considered as "normal" ligatures, which are enabled by default.

## Fonts Modification with [FontForge](https://github.com/fontforge/fontforge).

FontForge provide utilities to edit fonts. For our purpose, modification of some LUT (just called "Lookup" here) is needed. `Element -> FontInfo... -> Lookups` contains various tables. Table `liga` is the one for "normal" ligatures, and table `dlig` is the one for "ct" "st" ligatures.

There are already five rules in `liga`, which are:

```
ﬀ <- f f
ﬃ <- f f i
ﬄ <- f f l
ﬁ <- f i
ﬂ <- f l
```

I presume these are rules applied during the rendering process. Upon detection of corresponding inputs, combination of characters are replaced by the character on the left, which has their special render pattern, producing the ligature effect[^1]. By appending rules from `dlig` LUT, we now have seven rules:

```
ﬀ <- f f
ﬃ <- f f i
ﬄ <- f f l
ﬁ <- f i
ﬂ <- f l
c_t <- c t
s_t <- s t
```

("c_t" and "s_t" are notations for representing corresponding ligatures. It seems "ct" does not have its own Unicode character, whereas "st" does (`ﬆ`).)

After modification, we only need to export the modified fonts and use them in Kindle. Some extra cares should be taken, as FontForge did send several warnings while exporting fonts.

## Result

Following the process one can have the modified fonts installed on Kindle. The one I modified (used in this page) can be found on this page (tautology it is); pressing F12 and it is available somewhere.

## Acknowledgement

`kpfonts` is designed by [Christophe Caignaert](https://ctan.org/author/caignaert). `kpfonts-otf` is maintained by [Daniel Flipo](kpfonts-otf). I feel very lucky: thanks to their kindness, I am able to do some hacks to make these ligatures appear on my Kindle, despite knowing nothing about typography or font design.

----

[^1]: How the replacement process actually work is beyond this article's concern, but it does have some interesting patterns, rending ligatures like "fffi".