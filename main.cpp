// This is a quick C++ port of the JavaScript SW rasterizer by ryg:
// https://gist.github.com/2486101#file_raster3_ryg6.html

#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>

#include <immintrin.h>

#include <Windows.h>

using namespace std;

typedef unsigned char PixelComponent;
typedef unsigned int Pixel;

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))

#define USE_SIMD

class CApp {
private:
  static const int blocklog2 = 3;
  static const int blocksize = 1 << blocklog2;
  int canvasWidth;
  int canvasHeight;
  int canvasStride;
  int canvasWBlocks;
  int canvasHBlocks;
  PixelComponent *data;
  vector<unsigned char> block_state;

  static const int BLOCK_ISCLEAR    = 1 << 0;
  static const int BLOCK_NEEDCLEAR  = 1 << 1;
  static const int BLOCK_SOLID      = 1 << 2;

  static double random()
  {
    // Custom random number generator...
    static unsigned int seed1 = 0x23125253;
    static unsigned int seed2 = 0x7423e823;
    seed1 = 36969 * (seed1 & 65535) + (seed1 >> 16);
    seed2 = 18000 * (seed2 & 65535) + (seed2 >> 16);
    return (double)((seed1 << 16) + seed2) / (double)0xffffffffU;
  }


public:
  CApp()
  {
    canvasWidth = 1024;
    canvasHeight = 512;
    canvasStride = ((canvasWidth + 3) & ~3) * 4; // in bytes!
    canvasStride += 64; // increase stride to improve spread across cache sets
    data = (PixelComponent *)_aligned_malloc(canvasStride * canvasHeight, 64);
    memset(data, 0, canvasStride * canvasHeight);

    canvasWBlocks = (canvasWidth + blocksize-1) / blocksize;
    canvasHBlocks = (canvasHeight + blocksize-1) / blocksize;
    block_state.resize(canvasWBlocks * canvasHBlocks);
    memset(&block_state[0], BLOCK_ISCLEAR, block_state.size());
  }

  ~CApp()
  {
    _aligned_free(data);
  }

  void render()
  {
    // set block state for clearing
    for (int i=0; i < block_state.size(); i++)
      block_state[i] = (block_state[i] & BLOCK_ISCLEAR) ? BLOCK_ISCLEAR : BLOCK_NEEDCLEAR;

    // Draw 1000 triangles
    for (int i = 0; i < 100; i++) {
      drawTriangle(
            random() * canvasWidth,
            random() * canvasHeight,
            (Pixel)(random() * 0xffffff),
            random() * canvasWidth,
            random() * canvasHeight,
            (Pixel)(random() * 0xffffff),
            random() * canvasWidth,
            random() * canvasHeight,
            (Pixel)(random() * 0xffffff)
            );
    }

    // clear remaining blocks that need clearing
    performClear();
  }

  void performClear()
  {
    for (int y=0; y < canvasHBlocks; y++)
    {
      for (int x=0; x < canvasWBlocks; x++)
      {
        int ind = y*canvasWBlocks + x;
        if (block_state[ind] & BLOCK_NEEDCLEAR)
        {
          clearBlock(y*blocksize*canvasStride + x*blocksize*4);
          block_state[ind] = BLOCK_ISCLEAR;
        }
      }
    }
  }

  struct Edge
  {
    int dx, dy, offs;
    int dxline;
    int nmin, nmax;

    void setup(int x1, int y1, int x2, int y2, int minx, int miny, int block)
    {
      dx = x1 - x2;
      dy = y2 - y1;
      
      dxline = dx - block*dy;

      // offset
      offs = -dy*(x1 - minx) - dx*(y1 - miny);
      if (dy <= 0 && (dy != 0 || dx <= 0)) offs--;
      offs >>= 4;

      // min/max corners
      nmin = nmax = 0;
      if (dx >= 0) nmax -= dx; else nmin -= dx;
      if (dy >= 0) nmax -= dy; else nmin -= dy;
      nmin *= block-1;
      nmax *= block-1;
    }
  };

  void clearBlock(int offset)
  {
    unsigned char *ptr = &data[offset];
    for (int iy=0; iy < blocksize; iy++)
    {
      memset(ptr, 0, blocksize*4);
      ptr += canvasStride;
    }
  }

  struct SolidSetup
  {
    float fixscale;
    int dudx, dudy;
    int dvdx, dvdy;
#ifdef USE_SIMD
    __m128i mu_initstep, mv_initstep;
    __m128i deltas;
#endif

    void setup(const Edge *e, float scale)
    {
      fixscale = scale * 65536;
      dudx = (int) (e[0].dy * fixscale);
      dudy = (int) (e[0].dxline * fixscale);
      dvdx = (int) (e[1].dy * fixscale);
      dvdy = (int) (e[1].dxline * fixscale);

#ifdef USE_SIMD
      __m128i step0123 = _mm_set_epi32(3, 2, 1, 0);
      mu_initstep = _mm_mullo_epi32(step0123, _mm_set1_epi32(dudx));
      mv_initstep = _mm_mullo_epi32(step0123, _mm_set1_epi32(dvdx));
      deltas.m128i_i32[0] = dudx << 2;
      deltas.m128i_i32[1] = dvdx << 2;
      deltas.m128i_i32[2] = dudy;
      deltas.m128i_i32[3] = dvdy;
#endif
    }
  };

  template<bool masked>
  void solidBlock(int offset, int c1, int c2, int c3, const SolidSetup &s)
  {
    int u = (int) (c1 * s.fixscale);
    int v = (int) (c2 * s.fixscale);
    unsigned char *ptr = &data[offset];
    int linestep = canvasStride - blocksize*4;

#ifndef USE_SIMD
    for ( int iy = 0; iy < blocksize; iy ++ )
    {
      for ( int ix = 0; ix < blocksize; ix ++ )
      {
        if (!masked || !ptr[3])
        {
          ptr[0] = u >> 16;
          ptr[1] = v >> 16;
          ptr[2] = 0;
          ptr[3] = 255;
        }

        u += s.dudx;
        v += s.dvdx;
        ptr += 4;
      }

      u += s.dudy;
      v += s.dvdy;
      ptr += linestep;
    }
#else
    __m128i mu = _mm_add_epi32(_mm_set1_epi32(u), s.mu_initstep);
    __m128i mv = _mm_add_epi32(_mm_set1_epi32(v), s.mv_initstep);
    __m128i mdudx = _mm_shuffle_epi32(s.deltas, 0x00);
    __m128i mdvdx = _mm_shuffle_epi32(s.deltas, 0x55);
    __m128i mdudy = _mm_shuffle_epi32(s.deltas, 0xaa);
    __m128i mdvdy = _mm_shuffle_epi32(s.deltas, 0xff);

    for (int iy=0; iy < blocksize; iy++)
    {
      for (int ix=0; ix < blocksize/4; ix++)
      {
        __m128i ushift = _mm_srli_epi32(mu, 16);
        __m128i vshift = _mm_srli_epi32(mv, 16);

        __m128i pix0 = _mm_or_si128(ushift, _mm_slli_epi32(vshift, 8));
        __m128i pix1 = _mm_or_si128(pix0, _mm_set1_epi32(0xff000000));

        if (masked)
        {
          __m128i curpixel = _mm_load_si128((const __m128i*)ptr);
          __m128i mskip = _mm_srai_epi32(curpixel, 31); // alpha >= 128
          __m128i merged = _mm_blendv_epi8(pix1, curpixel, mskip);
          _mm_store_si128((__m128i *)ptr, merged);
        }
        else
          _mm_store_si128((__m128i *)ptr, pix1);

        mu = _mm_add_epi32(mu, mdudx);
        mv = _mm_add_epi32(mv, mdvdx);
        ptr += 4*4;
      }

      mu = _mm_add_epi32(mu, mdudy);
      mv = _mm_add_epi32(mv, mdvdy);
      ptr += linestep;
    }
#endif
  }

  void partialBlock(int offset, int c1, int c2, int c3, const Edge *e, float scale)
  {
    int pix1 = e[0].dy;
    int pix2 = e[1].dy;
    int pix3 = e[2].dy;
    int line1 = e[0].dxline;
    int line2 = e[1].dxline;
    int line3 = e[2].dxline;
    unsigned char *ptr = &data[offset];
    int linestep = canvasStride - blocksize*4;

#ifndef USE_SIMD
    for (int iy=0; iy < blocksize; iy++)
    {
      for (int ix=0; ix < blocksize; ix++)
      {
        if ((c1 | c2 | c3) >= 0 && !ptr[3])
        {
          int u = (int)(c1 * scale); // 0-255!
          int v = (int)(c2 * scale); // 0-255!
          ptr[0] = u;
          ptr[1] = v;
          ptr[2] = 0;
          ptr[3] = 255;
        }

        c1 += pix1;
        c2 += pix2;
        c3 += pix3;
        ptr += 4;
      }

      c1 += line1;
      c2 += line2;
      c3 += line3;
      ptr += linestep;
    }
#else
    __m128i mc1 = _mm_set1_epi32(c1);
    __m128i mc2 = _mm_set1_epi32(c2);
    __m128i mc3 = _mm_set1_epi32(c3);
    __m128i mpix1 = _mm_set1_epi32(pix1);
    __m128i mpix2 = _mm_set1_epi32(pix2);
    __m128i mpix3 = _mm_set1_epi32(pix3);
    __m128i mline1 = _mm_set1_epi32(line1);
    __m128i mline2 = _mm_set1_epi32(line2);
    __m128i mline3 = _mm_set1_epi32(line3);

    __m128i step0123 = _mm_set_epi32(3, 2, 1, 0);
    mc1 = _mm_add_epi32(mc1, _mm_mullo_epi32(mpix1, step0123));
    mc2 = _mm_add_epi32(mc2, _mm_mullo_epi32(mpix2, step0123));
    mc3 = _mm_add_epi32(mc3, _mm_mullo_epi32(mpix3, step0123));
    mpix1 = _mm_slli_epi32(mpix1, 2);
    mpix2 = _mm_slli_epi32(mpix2, 2);
    mpix3 = _mm_slli_epi32(mpix3, 2);

    // we can convert these to floating point. this is exact for fairly subtle reasons.
    __m128 mc1f = _mm_cvtepi32_ps(mc1);
    __m128 mc2f = _mm_cvtepi32_ps(mc2);
    __m128 mpix1f = _mm_cvtepi32_ps(mpix1);
    __m128 mpix2f = _mm_cvtepi32_ps(mpix2);
    __m128 mline1f = _mm_cvtepi32_ps(mline1);
    __m128 mline2f = _mm_cvtepi32_ps(mline2);

    __m128 mscale = _mm_set1_ps(scale);

    for (int iy=0; iy < blocksize; iy++)
    {
      for (int ix=0; ix < blocksize/4; ix++)
      {
        __m128i curpixel = _mm_load_si128((const __m128i*)ptr);
        __m128i csigns = _mm_or_si128(_mm_castps_si128(_mm_or_ps(mc1f, mc2f)), _mm_or_si128(mc3, curpixel));
        __m128i mskip = _mm_srai_epi32(csigns, 31);

        __m128 c1fs = _mm_mul_ps(mc1f, mscale);
        __m128 c2fs = _mm_mul_ps(mc2f, mscale);
        __m128i mu = _mm_cvttps_epi32(c1fs);
        __m128i mv = _mm_cvttps_epi32(c2fs);

        __m128i pix0 = _mm_or_si128(mu, _mm_slli_epi32(mv, 8));
        __m128i pix1 = _mm_or_si128(pix0, _mm_set1_epi32(0xff000000));

        __m128i merged = _mm_blendv_epi8(pix1, curpixel, mskip);
        _mm_store_si128((__m128i *)ptr, merged);

        mc1f = _mm_add_ps(mc1f, mpix1f);
        mc2f = _mm_add_ps(mc2f, mpix2f);
        mc3 = _mm_add_epi32(mc3, mpix3);
        ptr += 4*4;
      }

      mc1f = _mm_add_ps(mc1f, mline1f);
      mc2f = _mm_add_ps(mc2f, mline2f);
      mc3 = _mm_add_epi32(mc3, mline3);
      ptr += linestep;
    }
#endif
  }

  void drawTriangle( double ax1, double ay1, Pixel color1, double ax2, double ay2, Pixel color2, double ax3, double ay3, Pixel color3 )
  {
    // http://devmaster.net/forums/topic/1145-advanced-rasterization/

    // 28.4 fixed-point coordinates
    int x1 = (int)( 16.0 * ax1 + 0.5 );
    int x2 = (int)( 16.0 * ax2 + 0.5 );
    int x3 = (int)( 16.0 * ax3 + 0.5 );

    int y1 = (int)( 16.0 * ay1 + 0.5 );
    int y2 = (int)( 16.0 * ay2 + 0.5 );
    int y3 = (int)( 16.0 * ay3 + 0.5 );

    // Bounding rectangle
    int minx = MAX( ( MIN( x1, MIN ( x2, x3 ) ) + 0xf ) >> 4, 0 );
    int maxx = MIN( ( MAX( x1, MAX ( x2, x3 ) ) + 0xf ) >> 4, canvasWidth );
    int miny = MAX( ( MIN( y1, MIN ( y2, y3 ) ) + 0xf ) >> 4, 0 );
    int maxy = MIN( ( MAX( y1, MAX ( y2, y3 ) ) + 0xf ) >> 4, canvasHeight );

    // Block size, standard 8x8 (must be power of two)
    int q = blocksize;

    // Start in corner of 8x8 block
    minx &= ~(q - 1);
    miny &= ~(q - 1);

    // Edges
    Edge e[3];
    e[0].setup(x1, y1, x2, y2, minx << 4, miny << 4, q);
    e[1].setup(x2, y2, x3, y3, minx << 4, miny << 4, q);
    e[2].setup(x3, y3, x1, y1, minx << 4, miny << 4, q);

    // Block setup
    float scale = 255.0f / (e[0].offs + e[1].offs + e[2].offs);
    SolidSetup solid;
    solid.setup(e, scale);

    // Loop through blocks
    int linestep = canvasStride - q*4;
    int blockyind = (miny >> blocklog2) * canvasWBlocks;
    int cb1 = e[0].offs;
    int cb2 = e[1].offs;
    int cb3 = e[2].offs;
    int qstep = -q;
    int e1x = qstep*e[0].dy;
    int e2x = qstep*e[1].dy;
    int e3x = qstep*e[2].dy;
    int x0 = minx;

    for (int y0=miny; y0 < maxy; y0 += q)
    {
      // new block line - keep hunting for tri outer edge in old block line dir
      while (x0 >= minx && x0 < maxx && cb1 >= e[0].nmax && cb2 >= e[1].nmax && cb3 >= e[2].nmax)
      {
        x0 += qstep;
        cb1 += e1x;
        cb2 += e2x;
        cb3 += e3x;
      }

      // okay, we're now in a block we know is outside. reverse direction and go into main loop.
      qstep = -qstep;
      e1x = -e1x;
      e2x = -e2x;
      e3x = -e3x;
      for (;;)
      {
        x0 += qstep;
        cb1 += e1x;
        cb2 += e2x;
        cb3 += e3x;

        // We're done with this block row when at least one edge completely out
        if (x0 < minx || x0 >= maxx)
          break;

        // Skip block when at least one edge completely out
        // If an edge function is too small and decreasing in the current traversal dir,
        // we're done with this line.
        if (cb1 < e[0].nmax) if (e1x < 0) break; else continue;
        if (cb2 < e[1].nmax) if (e2x < 0) break; else continue;
        if (cb3 < e[2].nmax) if (e3x < 0) break; else continue;

        // We can skip this block if it's already fully covered
        int blockInd = blockyind + (x0 >> blocklog2);
        int state = block_state[blockInd];
        if (state & BLOCK_SOLID)
          continue;

        // Offset at top-left corner
        int offset = x0*4 + y0*canvasStride;

        // Accept whole block when fully covered
        if (cb1 >= e[0].nmin && cb2 >= e[1].nmin && cb3 >= e[2].nmin)
        {
          block_state[blockInd] = BLOCK_SOLID;
          if (state & BLOCK_NEEDCLEAR)
            solidBlock<false>(offset, cb1, cb2, cb3, solid);
          else
            solidBlock<true>(offset, cb1, cb2, cb3, solid);
        }
        else
        {
          block_state[blockInd] = state & ~(BLOCK_ISCLEAR | BLOCK_NEEDCLEAR);
          if (state & BLOCK_NEEDCLEAR)
            clearBlock(offset);
          partialBlock(offset, cb1, cb2, cb3, e, scale);
        }
      }
      
      // advance to next row of blocks
      cb1 += q*e[0].dx;
      cb2 += q*e[1].dx;
      cb3 += q*e[2].dx;
      blockyind += canvasWBlocks;
    }
  }

  void saveAsTGA(const char* fileName)
  {
    // Construct a TGA file header (18 bytes)
    unsigned char hdr[18];
    memset(hdr, 0, 18);
    hdr[2] = 2;     // True color
    hdr[12] = canvasWidth & 255;
    hdr[13] = canvasWidth >> 8;
    hdr[14] = canvasHeight & 255;
    hdr[15] = canvasHeight >> 8;
    hdr[16] = 32;   // 32 bpp

    // Write file
    ofstream of(fileName, ios_base::out | ios_base::binary);
    of.write((const char*)hdr, 18);
    for (int i=0; i < canvasHeight; i++)
      of.write((const char*)&data[i*canvasStride], canvasWidth*4);
    of.close();
  }
};

int main()
{
  static const int nFrames = 500;

  CApp app;

  LARGE_INTEGER tstart, tend, freq;
  QueryPerformanceFrequency(&freq);
  QueryPerformanceCounter(&tstart);

  // Benchmark the renderer
  for ( int i = 0; i < nFrames; i++ )
  {
    app.render();
  }

  QueryPerformanceCounter(&tend);
  printf("%.3fms / frame\n", 1000.0 * (tend.QuadPart - tstart.QuadPart) / (nFrames * freq.QuadPart));

  app.saveAsTGA("test.tga");

  return 0;
}