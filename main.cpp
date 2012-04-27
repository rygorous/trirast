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
	vector<bool> block_full;

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

		l = block_full.size();
		for ( i = 0; i < l; i++ )
		{
			block_full[i] = 0;
		}

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

    void setup(int x1, int y1, int x2, int y2)
    {
      dx = x1 - x2;
      dy = y2 - y1;
      
      offs = -dy*x1 - dx*y1;
      if (dy <= 0 && (dy != 0 || dx <= 0)) offs--;
      offs >>= 4;
    }
  };

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
    Edge e12, e23, e31;
    e12.setup(x1, y1, x2, y2);
    e23.setup(x2, y2, x3, y3);
    e31.setup(x3, y3, x1, y1);

		// Set up min/max corners
		int qm1 = q - 1; // for convenience
		int nmin1 = 0, nmax1 = 0;
		int nmin2 = 0, nmax2 = 0;
		int nmin3 = 0, nmax3 = 0;
		if (e12.dx >= 0) nmax1 -= qm1*e12.dx; else nmin1 -= qm1*e12.dx;
		if (e12.dy >= 0) nmax1 -= qm1*e12.dy; else nmin1 -= qm1*e12.dy;
		if (e23.dx >= 0) nmax2 -= qm1*e23.dx; else nmin2 -= qm1*e23.dx;
		if (e23.dy >= 0) nmax2 -= qm1*e23.dy; else nmin2 -= qm1*e23.dy;
		if (e31.dx >= 0) nmax3 -= qm1*e31.dx; else nmin3 -= qm1*e31.dx;
		if (e31.dy >= 0) nmax3 -= qm1*e31.dy; else nmin3 -= qm1*e31.dy;

		// Loop through blocks
		int linestep = (canvasWidth - q) * 4;
		double scale = 255.0 / (e12.offs + e23.offs + e31.offs);

    int cyb1 = e12.offs + e12.dx * miny + e12.dy * minx;
    int cyb2 = e23.offs + e23.dx * miny + e23.dy * minx;
    int cyb3 = e31.offs + e31.dx * miny + e31.dy * minx;

		for ( int y0 = miny; y0 < maxy; y0 += q, cyb1 += q*e12.dx, cyb2 += q*e23.dx, cyb3 += q*e31.dx )
		{
      int cxb1 = cyb1;
      int cxb2 = cyb2;
      int cxb3 = cyb3;

			for ( int x0 = minx; x0 < maxx; x0 += q, cxb1 += q*e12.dy, cxb2 += q*e23.dy, cxb3 += q*e31.dy )
			{
				// Edge functions at top-left corner
        int cy1 = cxb1;
        int cy2 = cxb2;
        int cy3 = cxb3;

				// Skip block when at least one edge completely out
				if (cy1 < nmax1 || cy2 < nmax2 || cy3 < nmax3)
					continue;

				// Skip writing full block if it's already fully covered
				int blockX = (x0 / q);
				int blockY = (y0 / q);
				int blockInd = blockX + blockY * canvasWBlocks;
				if (block_full[blockInd])
					continue;

				// Offset at top-left corner
				int offset = (x0 + y0 * canvasWidth) * 4;

				// Accept whole block when fully covered
				if (cy1 >= nmin1 && cy2 >= nmin2 && cy3 >= nmin3)
				{
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

							cx1 += e12.dy;
							cx2 += e23.dy;
							offset += 4;
						}

						cy1 += e12.dx;
						cy2 += e23.dx;
						offset += linestep;
					}

					block_full[blockInd] = true;
				}
				else
				{
					// Partially covered block
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

							cx1 += e12.dy;
							cx2 += e23.dy;
							cx3 += e31.dy;
							offset += 4;
						}

						cy1 += e12.dx;
						cy2 += e23.dx;
						cy3 += e31.dx;
						offset += linestep;
					}
				}
			}
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