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
    for (int i = 0; i < 1000; i++) {
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
    __m256i zero = _mm256_setzero_si256();

    for (int iy=0; iy < blocksize; iy++)
    {
      _mm256_store_si256((__m256i*)ptr, zero);
      ptr += canvasStride;
    }
  }

  struct SolidSetup
  {
    float fixscale;
    int dudx, dudy;
    int dvdx, dvdy;
#ifdef USE_SIMD
    float scale;
    float fdudy, fdvdy;
    __m256 mu_xstep, mv_xstep;
#endif

    void setup(const Edge *e, float scal)
    {
      fixscale = scal * 65536;
      dudx = (int) (e[0].dy * fixscale);
      dudy = (int) (e[0].dxline * fixscale);
      dvdx = (int) (e[1].dy * fixscale);
      dvdy = (int) (e[1].dxline * fixscale);

#ifdef USE_SIMD
      scale = scal;
      fdudy = e[0].dx * scale;
      fdvdy = e[1].dx * scale;

      __m256 e0dy = _mm256_cvtepi32_ps(_mm256_set1_epi32(e[0].dy));
      __m256 e1dy = _mm256_cvtepi32_ps(_mm256_set1_epi32(e[1].dy));
      __m256 steps = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
      steps = _mm256_mul_ps(steps, _mm256_broadcast_ss(&scale));
      mu_xstep = _mm256_mul_ps(e0dy, steps);
      mv_xstep = _mm256_mul_ps(e1dy, steps);
#endif
    }
  };

  template<bool masked>
  __forceinline void solidBlock(int offset, int c1, int c2, int c3, const SolidSetup &s)
  {
    unsigned char *ptr = &data[offset];
    int linestep = canvasStride - blocksize*4;

#ifndef USE_SIMD
    int u = (int) (c1 * s.fixscale);
    int v = (int) (c2 * s.fixscale);

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
    __m256 mu = _mm256_cvtepi32_ps(_mm256_set1_epi32(c1));
    __m256 mv = _mm256_cvtepi32_ps(_mm256_set1_epi32(c2));
    __m256 mscale = _mm256_broadcast_ss(&s.scale);
    mu = _mm256_mul_ps(mu, mscale);
    mv = _mm256_mul_ps(mv, mscale);
    mu = _mm256_add_ps(mu, s.mu_xstep);
    mv = _mm256_add_ps(mv, s.mv_xstep);
    __m256 mdudy = _mm256_broadcast_ss(&s.fdudy);
    __m256 mdvdy = _mm256_broadcast_ss(&s.fdvdy);

    for (int iy=0; iy < blocksize; iy++)
    {
      // okay, this one is a bit weird. we don't have access to AVX2 (integer) intrinsics, so
      // we build the (ARGB888) color using floats. ugh.
      __m256 vround = _mm256_floor_ps(mv);
      __m256 vrscaled = _mm256_mul_ps(vround, _mm256_set1_ps(256.0f));
      __m256 uv = _mm256_add_ps(mu, vrscaled);
      __m256i pix0 = _mm256_cvttps_epi32(uv);
      __m256 pix1 = _mm256_or_ps(_mm256_castsi256_ps(pix0), _mm256_castsi256_ps(_mm256_set1_epi32(0xff000000)));

      if (masked)
      {
        __m256 curpixel = _mm256_load_ps((const float*)ptr);
        __m256 merged = _mm256_blendv_ps(pix1, curpixel, curpixel);
        _mm256_store_ps((float *)ptr, merged);
      }
      else
        _mm256_store_ps((float *)ptr, pix1);

      mu = _mm256_add_ps(mu, mdudy);
      mv = _mm256_add_ps(mv, mdvdy);
      ptr += canvasStride;
    }
#endif
  }

  struct PartialSetup
  {
    int dy1, dy2, dy3;
    int dx1, dx2, dx3;
    float scale;
#ifdef USE_SIMD
    __m256 e1_xstep, e2_xstep, e3_xstep;
    float fdx1, fdx2, fdx3;
#endif

    void setup(const Edge *e, float scal)
    {
      dy1 = e[0].dy; dy2 = e[1].dy; dy3 = e[2].dy;
      dx1 = e[0].dxline; dx2 = e[1].dxline; dx3 = e[2].dxline;
      scale = scal;
#ifdef USE_SIMD
      fdx1 = (float)e[0].dx;
      fdx2 = (float)e[1].dx;
      fdx3 = (float)e[2].dx;

      __m256 mdy1 = _mm256_cvtepi32_ps(_mm256_set1_epi32(e[0].dy));
      __m256 mdy2 = _mm256_cvtepi32_ps(_mm256_set1_epi32(e[1].dy));
      __m256 mdy3 = _mm256_cvtepi32_ps(_mm256_set1_epi32(e[2].dy));

      __m256 steps = _mm256_set_ps(7, 6, 5, 4, 3, 2, 1, 0);
      e1_xstep = _mm256_mul_ps(steps, mdy1);
      e2_xstep = _mm256_mul_ps(steps, mdy2);
      e3_xstep = _mm256_mul_ps(steps, mdy3);
#endif
    }
  };

  __forceinline void partialBlock(int offset, int c1, int c2, int c3, const PartialSetup &s)
  {
    unsigned char *ptr = &data[offset];

#ifndef USE_SIMD
    int linestep = canvasStride - blocksize*4;

    for (int iy=0; iy < blocksize; iy++)
    {
      for (int ix=0; ix < blocksize; ix++)
      {
        if ((c1 | c2 | c3) >= 0 && !ptr[3])
        {
          int u = (int)(c1 * s.scale); // 0-255!
          int v = (int)(c2 * s.scale); // 0-255!
          ptr[0] = u;
          ptr[1] = v;
          ptr[2] = 0;
          ptr[3] = 255;
        }

        c1 += s.dy1;
        c2 += s.dy2;
        c3 += s.dy3;
        ptr += 4;
      }

      c1 += s.dx1;
      c2 += s.dx2;
      c3 += s.dx3;
      ptr += linestep;
    }
#else
    __m256 mc1 = _mm256_cvtepi32_ps(_mm256_set1_epi32(c1));
    __m256 mc2 = _mm256_cvtepi32_ps(_mm256_set1_epi32(c2));
    __m256 mc3 = _mm256_cvtepi32_ps(_mm256_set1_epi32(c3));

    mc1 = _mm256_add_ps(mc1, s.e1_xstep);
    mc2 = _mm256_add_ps(mc2, s.e2_xstep);
    mc3 = _mm256_add_ps(mc3, s.e3_xstep);

    __m256 mdx1 = _mm256_broadcast_ss(&s.fdx1);
    __m256 mdx2 = _mm256_broadcast_ss(&s.fdx2);
    __m256 mdx3 = _mm256_broadcast_ss(&s.fdx3);
    __m256 mscale = _mm256_broadcast_ss(&s.scale);

    for (int iy=0; iy < blocksize; iy++)
    {
      __m256 curpixel = _mm256_load_ps((const float*)ptr);
      __m256 signs = _mm256_or_ps(_mm256_or_ps(mc1, mc2), _mm256_or_ps(mc3, curpixel));

      __m256 us = _mm256_mul_ps(mc1, mscale);
      __m256 vs = _mm256_mul_ps(mc2, mscale);

      // similar trick as in solidBlock to build output pixel
      __m256 vround = _mm256_floor_ps(vs);
      __m256 vrscaled = _mm256_mul_ps(vround, _mm256_set1_ps(256.0f));
      __m256 ui = _mm256_castsi256_ps(_mm256_cvttps_epi32(us));
      __m256 vi = _mm256_castsi256_ps(_mm256_cvttps_epi32(vrscaled));
      __m256 pix0 = _mm256_or_ps(ui, _mm256_castsi256_ps(_mm256_set1_epi32(0xff000000)));
      __m256 pix1 = _mm256_or_ps(pix0, vi);

      __m256 merged = _mm256_blendv_ps(pix1, curpixel, signs);
      _mm256_store_ps((float *)ptr, merged);

      mc1 = _mm256_add_ps(mc1, mdx1);
      mc2 = _mm256_add_ps(mc2, mdx2);
      mc3 = _mm256_add_ps(mc3, mdx3);
      ptr += canvasStride;
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
    PartialSetup partial;
    solid.setup(e, scale);
    partial.setup(e, scale);

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
          partialBlock(offset, cb1, cb2, cb3, partial);
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