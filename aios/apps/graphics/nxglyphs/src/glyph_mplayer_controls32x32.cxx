/********************************************************************************************
 * apps/graphics/nxglyphs/src/glyph_mediaplayer32x32.cxx
 *
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.  The
 * ASF licenses this file to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance with the
 * License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.  See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 ********************************************************************************************/

/********************************************************************************************
 * Included Files
 ********************************************************************************************/

/* Automatically NuttX bitmap file. */
/* Generated from play_music.png by bitmap_converter.py. */

#include <nuttx/config.h>

#include <sys/types.h>
#include <stdint.h>
#include <stdbool.h>

#include <nuttx/nx/nxglib.h>
#include <nuttx/video/fb.h>
#include <nuttx/video/rgbcolors.h>

#include "graphics/nxwidgets/crlepalettebitmap.hxx"

#include "graphics/nxglyphs.hxx"


/********************************************************************************************
 * Pre-Processor Definitions
 ********************************************************************************************/

#define BITMAP_WIDTH 32
#define BITMAP_HEIGHT 32
#define BITMAP_PALETTESIZE 8

using namespace NXWidgets;

/* RGB24 (8-8-8) Colors */

static const nxwidget_pixel_t palette[BITMAP_PALETTESIZE] =
{
  CONFIG_NXGLYPHS_BACKGROUNDCOLOR,     MKRGB( 11, 24,108), MKRGB( 63, 90,192), MKRGB(121,136,250),
  MKRGB(224,234,244), MKRGB( 69, 80,149), MKRGB(127,169,239), MKRGB(152,174,207),
};

static const nxwidget_pixel_t hilight_palette[BITMAP_PALETTESIZE] =
{
  CONFIG_NXGLYPHS_BACKGROUNDCOLOR,     MKRGB( 61, 74,158), MKRGB(113,140,242), MKRGB(171,186,255),
  MKRGB(255,255,255), MKRGB(119,130,199), MKRGB(177,219,255), MKRGB(202,224,255),
};

/* Bitmap definition for the "Play" button */

static const SRlePaletteBitmapEntry play_bitmap[] =
{
  { 32,   0},                                                              /* Row 0 */
  { 14,   0}, {  4,   7}, { 14,   0},                                      /* Row 1 */
  { 10,   0}, { 12,   7}, { 10,   0},                                      /* Row 2 */
  {  8,   0}, {  3,   6}, {  9,   7}, {  3,   6}, {  1,   7}, {  8,   0},  /* Row 3 */
  {  7,   0}, {  7,   6}, {  4,   7}, {  7,   6}, {  7,   0},              /* Row 4 */
  {  6,   0}, {  3,   3}, { 13,   6}, {  4,   3}, {  6,   0},              /* Row 5 */
  {  5,   0}, {  6,   3}, {  9,   6}, {  7,   3}, {  5,   0},              /* Row 6 */
  {  4,   0}, {  1,   2}, { 22,   3}, {  1,   2}, {  4,   0},              /* Row 7 */
  {  3,   0}, {  1,   5}, { 23,   3}, {  1,   2}, {  1,   5}, {  3,   0},  /* Row 8 */
  {  3,   0}, {  3,   2}, {  5,   3}, {  1,   6}, { 13,   3}, {  4,   2},
  {  3,   0},                                                              /* Row 9 */
  {  2,   0}, {  1,   5}, {  6,   2}, {  2,   3}, {  2,   4}, {  1,   6},
  {  9,   3}, {  6,   2}, {  1,   5}, {  2,   0},                          /* Row 10 */
  {  2,   0}, {  1,   5}, {  8,   2}, {  1,   7}, {  3,   4}, {  1,   6},
  {  2,   2}, {  1,   3}, { 10,   2}, {  1,   5}, {  2,   0},              /* Row 11 */
  {  1,   0}, {  4,   5}, {  6,   2}, {  1,   7}, {  4,   4}, {  1,   7},
  {  1,   3}, {  8,   2}, {  4,   5}, {  2,   0},                          /* Row 12 */
  {  1,   0}, {  7,   5}, {  3,   2}, {  1,   7}, {  6,   4}, {  1,   7},
  {  3,   2}, {  8,   5}, {  2,   0},                                      /* Row 13 */
  {  1,   0}, {  9,   5}, {  1,   2}, {  1,   7}, {  8,   4}, {  1,   7},
  {  8,   5}, {  1,   1}, {  1,   5}, {  1,   0},                          /* Row 14 */
  {  1,   0}, {  4,   1}, {  6,   5}, {  1,   7}, { 10,   4}, {  1,   7},
  {  3,   5}, {  4,   1}, {  1,   5}, {  1,   0},                          /* Row 15 */
  {  1,   0}, {  8,   1}, {  2,   5}, {  1,   7}, { 10,   4}, {  1,   7},
  {  7,   1}, {  1,   5}, {  1,   0},                                      /* Row 16 */
  {  1,   0}, { 10,   1}, {  1,   7}, {  9,   4}, {  1,   5}, {  8,   1},
  {  1,   5}, {  1,   0},                                                  /* Row 17 */
  {  1,   0}, { 10,   1}, {  1,   7}, {  7,   4}, {  1,   5}, { 10,   1},
  {  2,   0},                                                              /* Row 18 */
  {  1,   0}, {  1,   5}, {  9,   1}, {  1,   7}, {  5,   4}, {  1,   7},
  { 12,   1}, {  2,   0},                                                  /* Row 19 */
  {  2,   0}, {  9,   1}, {  1,   7}, {  4,   4}, {  1,   5}, { 13,   1},
  {  2,   0},                                                              /* Row 20 */
  {  2,   0}, {  9,   1}, {  1,   7}, {  2,   4}, {  1,   7}, {  1,   2},
  { 13,   1}, {  1,   5}, {  2,   0},                                      /* Row 21 */
  {  2,   0}, {  1,   5}, {  8,   1}, {  1,   7}, {  1,   4}, {  1,   5},
  {  6,   2}, {  9,   1}, {  3,   0},                                      /* Row 22 */
  {  3,   0}, {  7,   1}, {  1,   2}, {  1,   5}, { 10,   2}, {  6,   1},
  {  1,   5}, {  3,   0},                                                  /* Row 23 */
  {  4,   0}, {  4,   1}, { 16,   2}, {  4,   1}, {  4,   0},              /* Row 24 */
  {  4,   0}, {  1,   5}, {  2,   1}, { 18,   2}, {  2,   1}, {  5,   0},  /* Row 25 */
  {  5,   0}, {  1,   5}, {  1,   1}, { 19,   2}, {  1,   5}, {  5,   0},  /* Row 26 */
  {  6,   0}, {  1,   5}, { 18,   2}, {  7,   0},                          /* Row 27 */
  {  8,   0}, { 15,   2}, {  1,   5}, {  8,   0},                          /* Row 28 */
  {  9,   0}, {  1,   5}, { 12,   2}, { 10,   0},                          /* Row 29 */
  { 12,   0}, {  8,   2}, { 12,   0},                                      /* Row 30 */
  { 32,   0},                                                              /* Row 31 */
};

const struct SRlePaletteBitmap NXWidgets::g_mplayerPlayBitmap =
{
  CONFIG_NXWIDGETS_BPP,
  CONFIG_NXWIDGETS_FMT,
  BITMAP_PALETTESIZE,
  BITMAP_WIDTH,
  BITMAP_HEIGHT,
  {palette, hilight_palette},
  play_bitmap
};

/* Bitmap definition for the "Pause" button */

static const SRlePaletteBitmapEntry pause_bitmap[] =
{
  { 32,   0},                                                              /* Row 0 */
  { 14,   0}, {  4,   7}, { 14,   0},                                      /* Row 1 */
  { 10,   0}, { 12,   7}, { 10,   0},                                      /* Row 2 */
  {  8,   0}, {  4,   6}, {  8,   7}, {  4,   6}, {  8,   0},              /* Row 3 */
  {  7,   0}, {  7,   6}, {  3,   7}, {  8,   6}, {  7,   0},              /* Row 4 */
  {  6,   0}, {  3,   3}, { 13,   6}, {  4,   3}, {  6,   0},              /* Row 5 */
  {  5,   0}, {  6,   3}, {  9,   6}, {  7,   3}, {  5,   0},              /* Row 6 */
  {  4,   0}, {  1,   2}, { 22,   3}, {  1,   2}, {  4,   0},              /* Row 7 */
  {  3,   0}, {  1,   5}, { 23,   3}, {  1,   2}, {  1,   5}, {  3,   0},  /* Row 8 */
  {  3,   0}, {  3,   2}, { 19,   3}, {  4,   2}, {  3,   0},              /* Row 9 */
  {  2,   0}, {  1,   5}, {  5,   2}, {  2,   3}, {  1,   6}, {  3,   4},
  {  3,   3}, {  1,   7}, {  2,   4}, {  1,   7}, {  2,   3}, {  6,   2},
  {  1,   5}, {  2,   0},                                                  /* Row 10 */
  {  2,   0}, {  1,   5}, {  7,   2}, {  1,   7}, {  3,   4}, {  1,   7},
  {  1,   2}, {  1,   3}, {  4,   4}, {  1,   3}, {  7,   2}, {  1,   5},
  {  2,   0},                                                              /* Row 11 */
  {  1,   0}, {  4,   5}, {  5,   2}, {  1,   7}, {  3,   4}, {  1,   7},
  {  1,   2}, {  1,   3}, {  4,   4}, {  1,   6}, {  4,   2}, {  4,   5},
  {  2,   0},                                                              /* Row 12 */
  {  1,   0}, {  8,   5}, {  1,   2}, {  1,   7}, {  3,   4}, {  1,   7},
  {  1,   2}, {  1,   5}, {  4,   4}, {  1,   5}, {  1,   2}, {  7,   5},
  {  2,   0},                                                              /* Row 13 */
  {  1,   0}, {  9,   5}, {  1,   7}, {  3,   4}, {  1,   7}, {  1,   2},
  {  1,   5}, {  4,   4}, {  8,   5}, {  1,   1}, {  1,   5}, {  1,   0},  /* Row 14 */
  {  1,   0}, {  4,   1}, {  5,   5}, {  1,   7}, {  3,   4}, {  1,   7},
  {  2,   5}, {  4,   4}, {  5,   5}, {  4,   1}, {  1,   5}, {  1,   0},  /* Row 15 */
  {  1,   0}, {  9,   1}, {  1,   7}, {  3,   4}, {  1,   7}, {  2,   5},
  {  4,   4}, {  1,   5}, {  8,   1}, {  1,   5}, {  1,   0},              /* Row 16 */
  {  1,   0}, {  9,   1}, {  1,   7}, {  3,   4}, {  1,   7}, {  1,   1},
  {  1,   5}, {  4,   4}, {  1,   5}, {  8,   1}, {  1,   5}, {  1,   0},  /* Row 17 */
  {  1,   0}, {  9,   1}, {  1,   7}, {  3,   4}, {  1,   7}, {  1,   1},
  {  1,   5}, {  4,   4}, {  1,   5}, {  8,   1}, {  2,   0},              /* Row 18 */
  {  1,   0}, {  1,   5}, {  8,   1}, {  1,   7}, {  3,   4}, {  1,   7},
  {  1,   1}, {  1,   5}, {  4,   4}, {  1,   5}, {  8,   1}, {  2,   0},  /* Row 19 */
  {  2,   0}, {  8,   1}, {  1,   7}, {  3,   4}, {  1,   7}, {  1,   1},
  {  1,   5}, {  4,   4}, {  1,   5}, {  8,   1}, {  2,   0},              /* Row 20 */
  {  2,   0}, {  8,   1}, {  1,   7}, {  3,   4}, {  1,   7}, {  1,   1},
  {  1,   2}, {  4,   4}, {  1,   5}, {  7,   1}, {  1,   5}, {  2,   0},  /* Row 21 */
  {  2,   0}, {  1,   5}, {  7,   1}, {  1,   7}, {  3,   4}, {  1,   6},
  {  1,   1}, {  1,   2}, {  4,   4}, {  1,   2}, {  7,   1}, {  3,   0},  /* Row 22 */
  {  3,   0}, {  7,   1}, {  1,   2}, {  3,   6}, {  4,   2}, {  2,   6},
  {  2,   2}, {  6,   1}, {  1,   5}, {  3,   0},                          /* Row 23 */
  {  4,   0}, {  4,   1}, { 16,   2}, {  4,   1}, {  4,   0},              /* Row 24 */
  {  4,   0}, {  1,   5}, {  2,   1}, { 18,   2}, {  2,   1}, {  5,   0},  /* Row 25 */
  {  5,   0}, {  1,   5}, {  1,   1}, {  6,   2}, {  6,   2}, {  7,   2},
  {  1,   5}, {  5,   0},                                                  /* Row 26 */
  {  6,   0}, {  1,   5}, {  4,   2}, {  1,   5}, {  8,   2}, {  1,   5},
  {  4,   2}, {  7,   0},                                                  /* Row 27 */
  {  8,   0}, {  3,   2}, {  4,   2}, {  2,   2}, {  4,   2}, {  2,   2},
  {  1,   5}, {  8,   0},                                                  /* Row 28 */
  {  9,   0}, {  1,   5}, {  3,   2}, {  6,   2}, {  3,   2}, { 10,   0},  /* Row 29 */
  { 12,   0}, {  8,   2}, { 12,   0},                                      /* Row 30 */
  { 32,   0},                                                              /* Row 31 */
};

const struct SRlePaletteBitmap NXWidgets::g_mplayerPauseBitmap =
{
  CONFIG_NXWIDGETS_BPP,
  CONFIG_NXWIDGETS_FMT,
  BITMAP_PALETTESIZE,
  BITMAP_WIDTH,
  BITMAP_HEIGHT,
  {palette, hilight_palette},
  pause_bitmap
};

/* Bitmap definition for "Rewind" control */

static const SRlePaletteBitmapEntry rew_bitmap[] =
{
  { 32,   0},                                                              /* Row 0 */
//  { 14,   0}, {  3,   4}, {  1,   7}, { 14,   0},                          /* Row 1 */
  { 14,   0}, {  4,   7}, { 14,   0},                                      /* Row 1 */
  { 10,   0}, { 12,   7}, { 10,   0},              /* Row 2 */
  {  8,   0}, {  1,   7}, {  3,   6}, {  8,   7},
  {  3,   6}, {  1,   7}, {  8,   0},                                      /* Row 3 */
  {  7,   0}, {  7,   6}, {  3,   7}, {  8,   6}, {  7,   0},              /* Row 4 */
  {  6,   0}, {  3,   3}, { 13,   6}, {  4,   3}, {  6,   0},              /* Row 5 */
  {  5,   0}, {  6,   3}, {  9,   6}, {  7,   3}, {  5,   0},              /* Row 6 */
  {  4,   0}, { 23,   3}, {  1,   2}, {  4,   0},                          /* Row 7 */
  {  3,   0}, {  1,   5}, { 23,   3}, {  1,   2}, {  1,   5}, {  3,   0},  /* Row 8 */
  {  3,   0}, {  3,   2}, { 18,   3}, {  5,   2}, {  3,   0},              /* Row 9 */
  {  2,   0}, {  1,   5}, {  5,   2}, {  6,   3}, {  1,   6}, {  7,   3},
  {  2,   2}, {  2,   6}, {  3,   2}, {  1,   5}, {  2,   0},              /* Row 10 */
  {  2,   0}, {  1,   5}, {  9,   2}, {  1,   6}, {  2,   4}, {  1,   3},
  {  1,   2}, {  2,   3}, {  3,   2}, {  1,   6}, {  2,   4}, {  1,   6},
  {  3,   2}, {  1,   5}, {  2,   0},                                      /* Row 11 */
  {  1,   0}, {  2,   5}, {  7,   2}, {  1,   6}, {  4,   4}, {  1,   3},
  {  4,   2}, {  1,   3}, {  1,   7}, {  3,   4}, {  1,   7}, {  3,   2},
  {  1,   5}, {  2,   0},                                                  /* Row 12 */
  {  1,   0}, {  5,   5}, {  2,   2}, {  1,   5}, {  1,   7}, {  5,   4},
  {  1,   5}, {  3,   2}, {  1,   7}, {  5,   4}, {  1,   7}, {  1,   2},
  {  3,   5}, {  2,   0},                                                  /* Row 13 */
  {  1,   0}, {  6,   5}, {  1,   7}, {  7,   4}, {  1,   5}, {  1,   2},
  {  1,   7}, {  7,   4}, {  1,   7}, {  3,   5}, {  1,   1}, {  1,   5},
  {  1,   0},                                                              /* Row 14 */
  {  1,   0}, {  4,   1}, {  1,   7}, {  9,   4}, {  1,   7}, {  9,   4},
  {  1,   7}, {  4,   1}, {  1,   5}, {  1,   0},                          /* Row 15 */
  {  1,   0}, {  4,   1}, {  1,   5}, {  9,   4}, {  1,   7}, {  9,   4},
  {  1,   5}, {  4,   1}, {  1,   5}, {  1,   0},                          /* Row 16 */
  {  1,   0}, {  6,   1}, {  1,   5}, {  7,   4}, {  1,   5}, {  1,   1},
  {  1,   5}, {  7,   4}, {  1,   5}, {  4,   1}, {  1,   5}, {  1,   0},  /* Row 17 */
  {  1,   0}, {  8,   1}, {  1,   7}, {  5,   4}, {  1,   5}, {  3,   1},
  {  1,   7}, {  5,   4}, {  1,   5}, {  4,   1}, {  2,   0},              /* Row 18 */
  {  1,   0}, {  1,   5}, {  8,   1}, {  1,   5}, {  4,   4}, {  1,   5},
  {  4,   1}, {  1,   5}, {  4,   4}, {  1,   5}, {  4,   1}, {  2,   0},  /* Row 19 */
  {  2,   0}, { 10,   1}, {  1,   7}, {  2,   4}, {  1,   2}, {  6,   1},
  {  1,   7}, {  2,   4}, {  1,   5}, {  4,   1}, {  2,   0},              /* Row 20 */
  {  2,   0}, { 11,   1}, {  1,   5}, {  1,   4}, {  1,   2}, {  7,   1},
  {  1,   5}, {  1,   7}, {  1,   5}, {  3,   1}, {  1,   5}, {  2,   0},  /* Row 21 */
  {  2,   0}, {  1,   5}, {  9,   1}, {  1,   2}, {  1,   1}, {  6,   2},
  {  9,   1}, {  3,   0},                                                  /* Row 22 */
  {  3,   0}, {  7,   1}, { 12,   2}, {  6,   1}, {  1,   5}, {  3,   0},  /* Row 23 */
  {  4,   0}, {  5,   1}, { 14,   2}, {  5,   1}, {  4,   0},              /* Row 24 */
  {  4,   0}, {  1,   5}, {  2,   1}, { 18,   2}, {  2,   1}, {  5,   0},  /* Row 25 */
  {  5,   0}, {  1,   5}, {  1,   1}, {  6,   2}, {  6,   2}, {  7,   2},
  {  1,   5}, {  5,   0},                                                  /* Row 26 */
  {  6,   0}, {  1,   5}, {  4,   2}, {  1,   5}, {  8,   2}, {  1,   5},
  {  4,   2}, {  7,   0},                                                  /* Row 27 */
  {  8,   0}, {  3,   2}, {  3,   2}, {  4,   2}, {  3,   2}, {  2,   2},
  {  1,   5}, {  8,   0},                                                  /* Row 28 */
  {  9,   0}, {  1,   5}, {  3,   2}, {  6,   2}, {  2,   2}, {  1,   5},
  { 10,   0},                                                              /* Row 29 */
  { 12,   0}, {  3,   2}, {  2,   2}, {  3,   2}, { 12,   0},              /* Row 30 */
  { 32,   0},                                                              /* Row 31 */
};

const struct SRlePaletteBitmap NXWidgets::g_mplayerRewBitmap =
{
  CONFIG_NXWIDGETS_BPP,
  CONFIG_NXWIDGETS_FMT,
  BITMAP_PALETTESIZE,
  BITMAP_WIDTH,
  BITMAP_HEIGHT,
  {palette, hilight_palette},
  rew_bitmap
};

/* Bitmap definition for "Forward" control */

static const SRlePaletteBitmapEntry fwd_bitmap[] =
{
  { 32,   0},                                                              /* Row 0 */
  { 14,   0}, {  4,   7}, { 14,   0},                                      /* Row 1 */
  { 10,   0}, { 12,   7}, { 10,   0},                                      /* Row 2 */
  {  8,   0}, {  1,   7}, {  3,   6}, {  8,   7},
  {  3,   6}, {  1,   7}, {  8,   0},                                      /* Row 3 */
  {  7,   0}, {  7,   6}, {  3,   7}, {  8,   6}, {  7,   0},              /* Row 4 */
  {  6,   0}, {  3,   3}, { 13,   6}, {  4,   3}, {  6,   0},              /* Row 5 */
  {  5,   0}, {  6,   3}, { 10,   6}, {  6,   3}, {  5,   0},              /* Row 6 */
  {  4,   0}, {  1,   2}, { 22,   3}, {  1,   2}, {  4,   0},              /* Row 7 */
  {  3,   0}, {  1,   5}, { 23,   3}, {  1,   2}, {  1,   5}, {  3,   0},  /* Row 8 */
  {  3,   0}, {  3,   2}, { 19,   3}, {  4,   2}, {  3,   0},              /* Row 9 */
  {  2,   0}, {  1,   5}, {  3,   2}, {  2,   6}, {  2,   2}, {  7,   3},
  {  1,   6}, {  5,   3}, {  6,   2}, {  1,   5}, {  2,   0},              /* Row 10 */
  {  2,   0}, {  1,   5}, {  3,   2}, {  1,   7}, {  2,   4}, {  1,   3},
  {  2,   2}, {  3,   3}, {  1,   2}, {  1,   6}, {  2,   4}, {  1,   6},
  {  9,   2}, {  1,   5}, {  2,   0},                                      /* Row 11 */
  {  1,   0}, {  2,   5}, {  3,   2}, {  1,   7}, {  3,   4}, {  1,   7},
  {  5,   2}, {  1,   6}, {  4,   4}, {  1,   5}, {  7,   2}, {  1,   5},
  {  2,   0},                                                              /* Row 12 */
  {  1,   0}, {  4,   5}, {  1,   2}, {  1,   7}, {  5,   4}, {  1,   7},
  {  3,   2}, {  1,   6}, {  5,   4}, {  1,   7}, {  1,   5}, {  2,   2},
  {  4,   5}, {  2,   0},                                                  /* Row 13 */
  {  1,   0}, {  1,   5}, {  1,   1}, {  2,   5}, {  1,   1}, {  1,   7},
  {  7,   4}, {  1,   7}, {  1,   2}, {  1,   5}, {  7,   4}, {  1,   7},
  {  4,   5}, {  1,   1}, {  1,   5}, {  1,   0},                          /* Row 14 */
  {  1,   0}, {  5,   1}, {  1,   7}, {  9,   4}, {  1,   7}, {  9,   4},
  {  1,   5}, {  3,   1}, {  1,   5}, {  1,   0},                          /* Row 15 */
  {  1,   0}, {  5,   1}, {  1,   7}, {  8,   4}, {  2,   7}, {  8,   4},
  {  1,   7}, {  4,   1}, {  1,   5}, {  1,   0},                          /* Row 16 */
  {  1,   0}, {  5,   1}, {  1,   5}, {  7,   4}, {  1,   5}, {  1,   1},
  {  1,   5}, {  7,   4}, {  1,   5}, {  5,   1}, {  1,   5}, {  1,   0},  /* Row 17 */
  {  1,   0}, {  5,   1}, {  1,   5}, {  5,   4}, {  1,   7}, {  3,   1},
  {  1,   5}, {  5,   4}, {  1,   7}, {  7,   1}, {  2,   0},              /* Row 18 */
  {  1,   0}, {  1,   5}, {  4,   1}, {  1,   5}, {  3,   4}, {  1,   7},
  {  1,   2}, {  4,   1}, {  1,   5}, {  4,   4}, {  1,   5}, {  8,   1},
  {  2,   0},                                                              /* Row 19 */
  {  2,   0}, {  4,   1}, {  1,   5}, {  2,   4}, {  1,   5}, {  6,   1},
  {  1,   5}, {  2,   4}, {  1,   7}, { 10,   1}, {  2,   0},              /* Row 20 */
  {  2,   0}, {  4,   1}, {  2,   7}, {  8,   1}, {  1,   5}, {  1,   4},
  {  1,   2}, { 10,   1}, {  1,   5}, {  2,   0},                          /* Row 21 */
  {  2,   0}, {  1,   5}, {  9,   1}, {  6,   2}, {  1,   1}, {  1,   2},
  {  9,   1}, {  3,   0},                                                  /* Row 22 */
  {  3,   0}, {  7,   1}, { 12,   2}, {  6,   1}, {  1,   5}, {  3,   0},  /* Row 23 */
  {  4,   0}, {  5,   1}, { 14,   2}, {  5,   1}, {  4,   0},              /* Row 24 */
  {  4,   0}, {  1,   5}, {  2,   1}, { 18,   2}, {  2,   1}, {  5,   0},  /* Row 25 */
  {  5,   0}, {  1,   5}, {  1,   1}, {  6,   2}, {  6,   2}, {  7,   2},
  {  1,   5}, {  5,   0},                                                  /* Row 26 */
  {  6,   0}, {  1,   5}, {  4,   2}, {  1,   5}, {  8,   2}, {  1,   5},
  {  4,   2}, {  7,   0},                                                  /* Row 27 */
  {  8,   0}, {  3,   2}, {  3,   2}, {  4,   2}, {  3,   2}, {  2,   2},
  {  1,   5}, {  8,   0},                                                  /* Row 28 */
  {  9,   0}, {  1,   5}, {  3,   2}, {  6,   2}, {  2,   2}, {  1,   5},
  { 10,   0},                                                              /* Row 29 */
  { 12,   0}, {  3,   2}, {  2,   2}, {  3,   2}, { 12,   0},              /* Row 30 */
  { 32,   0},                                                              /* Row 31 */
};

const struct SRlePaletteBitmap NXWidgets::g_mplayerFwdBitmap =
{
  CONFIG_NXWIDGETS_BPP,
  CONFIG_NXWIDGETS_FMT,
  BITMAP_PALETTESIZE,
  BITMAP_WIDTH,
  BITMAP_HEIGHT,
  {palette, hilight_palette},
  fwd_bitmap
};

static const SRlePaletteBitmapEntry vol_bitmap[] =
{
  { 22,   0},                                                              /* Row 0 */
  {  9,   0}, {  5,   7}, {  8,   0},                                      /* Row 1 */
  {  6,   0}, {  2,   6}, {  6,   7}, {  2,   6}, {  1,   7}, {  5,   0},  /* Row 2 */
  {  4,   0}, {  2,   3}, {  9,   6}, {  3,   3}, {  4,   0},              /* Row 3 */
  {  3,   0}, {  5,   3}, {  6,   6}, {  5,   3}, {  3,   0},              /* Row 4 */
  {  3,   0}, { 16,   3}, {  3,   0},                                      /* Row 5 */
  {  2,   0}, {  2,   2}, { 13,   3}, {  3,   2}, {  2,   0},              /* Row 6 */
  {  1,   0}, {  1,   5}, {  4,   2}, { 10,   3}, {  4,   2}, {  1,   5},
  {  1,   0},                                                              /* Row 7 */
  {  1,   0}, {  2,   5}, { 15,   2}, {  3,   5}, {  1,   0},              /* Row 8 */
  {  1,   0}, {  5,   5}, { 10,   2}, {  5,   5}, {  1,   0},              /* Row 9 */
  {  1,   0}, {  2,   1}, { 15,   5}, {  3,   1}, {  1,   0},              /* Row 10 */
  {  1,   0}, {  5,   1}, { 10,   5}, {  5,   1}, {  1,   0},              /* Row 11 */
  {  1,   0}, { 20,   1}, {  1,   0},                                      /* Row 12 */
  {  1,   0}, { 20,   1}, {  1,   0},                                      /* Row 13 */
  {  1,   0}, { 19,   1}, {  1,   5}, {  1,   0},                          /* Row 14 */
  {  1,   0}, {  1,   5}, {  7,   1}, {  1,   5}, {  4,   2}, {  6,   1},
  {  2,   0},                                                              /* Row 15 */
  {  2,   0}, {  5,   1}, {  8,   2}, {  4,   1}, {  1,   5}, {  2,   0},  /* Row 16 */
  {  3,   0}, {  2,   1}, { 12,   2}, {  2,   1}, {  3,   0},              /* Row 17 */
  {  3,   0}, {  1,   5}, {  1,   1}, { 13,   2}, {  1,   5}, {  3,   0},  /* Row 18 */
  {  6,   0}, { 10,   2}, {  1,   5}, {  5,   0},                          /* Row 19 */
  {  6,   0}, {  1,   5}, {  8,   2}, {  7,   0},                          /* Row 20 */
  { 22,   0},                                                              /* Row 21 */
};

const struct SRlePaletteBitmap NXWidgets::g_mplayerVolBitmap =
{
  CONFIG_NXWIDGETS_BPP,
  CONFIG_NXWIDGETS_FMT,
  BITMAP_PALETTESIZE,
  22,
  22,
  {hilight_palette, hilight_palette},
  vol_bitmap
};
