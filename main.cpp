// This is a quick C++ port of the JavaScript SW rasterizer by ryg:
// https://gist.github.com/2486101#file_raster3_ryg6.html

#include <vector>
#include <cstring>
#include <iostream>
#include <fstream>

#include <Windows.h>

using namespace std;

typedef unsigned char PixelComponent;
typedef unsigned int Pixel;

#define MIN(x,y) ((x)<(y)?(x):(y))
#define MAX(x,y) ((x)>(y)?(x):(y))


class CApp {
private:
	static const int blocksize = 8;
	int canvasWidth;
	int canvasHeight;
	int canvasWBlocks;
	int canvasHBlocks;
	vector<PixelComponent> data;
	vector<unsigned char> block_full;

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
		data.resize(canvasWidth * canvasHeight * 4);

		canvasWBlocks = (canvasWidth + blocksize-1) / blocksize;
		canvasHBlocks = (canvasHeight + blocksize-1) / blocksize;
		block_full.resize(canvasWBlocks * canvasHBlocks);
	}

	void render()
	{
		int i, l;

		// clear
		l = data.size();

		for ( i = 3; i < l; i += 4 )
		{
			data[ i ] = 0;
		}

    memset(&block_full[0], 0, block_full.size() * sizeof(block_full[0]));

		// Draw 1000 triangles
		for ( i = 0; i < 1000; i++ ) {
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
	}

	void drawPixel( int x, int y, PixelComponent r, PixelComponent g, PixelComponent b )
	{
		int offset = ( x + y * canvasWidth ) * 4;

		if ( data[ offset + 3 ] )
			return;

		data[ offset ] = r;
		data[ offset + 1 ] = g;
		data[ offset + 2 ] = b;
		data[ offset + 3 ] = 255;
	}

	void drawRectangle( double x1, double y1, double x2, double y2, Pixel color )
	{
		PixelComponent r = color >> 16 & 255;
		PixelComponent g = color >> 8 & 255;
		PixelComponent b = color & 255;

		int xmin = (int)(MIN( x1, x2 ));
		int xmax = (int)(MAX( x1, x2 ));
		int ymin = (int)(MIN( y1, y2 ));
		int ymax = (int)(MAX( y1, y2 ));

		for ( int y = ymin; y < ymax; y ++ )
		{
			for ( int x = xmin; x < xmax; x ++ )
			{
				drawPixel( x, y, r, g, b );
			}
		}
	}

  struct Edge
  {
    int dx, dy, offs;
    int nmin, nmax;

    void setup(int x1, int y1, int x2, int y2, int minx, int miny, int block)
    {
      dx = x1 - x2;
      dy = y2 - y1;
      
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

  void doBlock(int x0, int y0, int cy1, int cy2, int cy3, const Edge *e, float scale)
  {
    int q = blocksize;

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

		// Loop through blocks
		int linestep = (canvasWidth - q) * 4;
		float scale = 255.0f / (e[0].offs + e[1].offs + e[2].offs);

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
		    int blockX = (x0 / q);
		    int blockY = (y0 / q);
		    int blockInd = blockX + blockY * canvasWBlocks;
		    if (block_full[blockInd])
          continue;

		    // Offset at top-left corner
		    int offset = (x0 + y0 * canvasWidth) * 4;

		    // Accept whole block when fully covered
		    if (cb1 >= e[0].nmin && cb2 >= e[1].nmin && cb3 >= e[2].nmin)
		    {
          int cy1 = cb1;
          int cy2 = cb2;

			    for ( int iy = 0; iy < q; iy ++ )
			    {
				    int cx1 = cy1;
				    int cx2 = cy2;

				    for ( int ix = 0; ix < q; ix ++ )
				    {
					    if (!data[offset + 3])
					    {
						    int u = (int)(cx1 * scale); // 0-255!
						    int v = (int)(cx2 * scale); // 0-255!
						    data[offset] = u;
						    data[offset + 1] = v;
						    data[offset + 2] = 0;
						    data[offset + 3] = 255;
					    }

					    cx1 += e[0].dy;
					    cx2 += e[1].dy;
					    offset += 4;
				    }

				    cy1 += e[0].dx;
				    cy2 += e[1].dx;
				    offset += (canvasWidth - q) * 4;
			    }

			    block_full[blockInd] = true;
		    }
		    else
		    {
			    // Partially covered block
          int cy1 = cb1;
          int cy2 = cb2;
          int cy3 = cb3;

			    for ( int iy = 0; iy < q; iy ++ )
			    {
				    int cx1 = cy1;
				    int cx2 = cy2;
				    int cx3 = cy3;

				    for ( int ix = 0; ix < q; ix ++ )
				    {
					    if ( (cx1 | cx2 | cx3) >= 0 && !data[offset+3])
					    {
						    int u = (int)(cx1 * scale); // 0-255!
						    int v = (int)(cx2 * scale); // 0-255!
						    data[offset] = u;
						    data[offset + 1] = v;
						    data[offset + 2] = 0;
						    data[offset + 3] = 255;
					    }

					    cx1 += e[0].dy;
					    cx2 += e[1].dy;
					    cx3 += e[2].dy;
					    offset += 4;
				    }

				    cy1 += e[0].dx;
				    cy2 += e[1].dx;
				    cy3 += e[2].dx;
				    offset += (canvasWidth - q) * 4;
			    }
		    }
      }
      
      // advance to next row of blocks
      cb1 += q*e[0].dx;
      cb2 += q*e[1].dx;
      cb3 += q*e[2].dx;
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
		of.write((const char*)&data[0], data.size());
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