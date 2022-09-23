// 
// Music visualizer using CUDA Discrete Fourier Transform
//
// 2022, Jonathan Tainer
//

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <raylib.h>

#include "cudaft.h"

#define FT_BUFFER_SIZE 512	// num of samples for fourier transform
#define SMOOTHING_CONST 0.9f	// 0.f == no smoothing, 1.f = max smoothing (no movement)

void UpdateFourierBuffer(float* dest, float* source, unsigned int destBufferSize, unsigned int sourceBufferSize, unsigned int channels, unsigned int sampleRate, float elapsedTime) {
	// Compute index to start sampling at
	unsigned int beginIndex = floorf(elapsedTime * sampleRate);

	for (int i = 0; i < destBufferSize; i++) {
		
		// If index is outside range of source buffer, fill dest with zeros
		if (i + beginIndex >= sourceBufferSize)
			dest[i] = 0.f;
		else
			dest[i] = source[(i + beginIndex) * channels];
	}
}

int main(int argc, char** argv) {

	// Get command16ine input
	if (argc < 2) {
		printf("No files specified\n");
		return 0;
	}

	// Window setup
	const int screenWidth = 1920;
	const int screenHeight = 1080;
	SetConfigFlags(FLAG_WINDOW_ALWAYS_RUN);
	InitWindow(screenWidth, screenHeight, argv[1]);
	SetTargetFPS(60);
//	ToggleFullscreen();
	
	// Load waveform data
	InitAudioDevice();
	Wave wave = LoadWave(argv[1]);
	Sound sound = LoadSoundFromWave(wave);

	// Convert waveform to floating point
	float* samples = LoadWaveSamples(wave);

	// Setup buffers and DFT matrix
	float sampleBuffer[FT_BUFFER_SIZE];
	float outputBuffer[FT_BUFFER_SIZE];
	float resultBuffer[FT_BUFFER_SIZE];
	memset(sampleBuffer, 0, sizeof(float) * FT_BUFFER_SIZE);
	memset(sampleBuffer, 0, sizeof(float) * FT_BUFFER_SIZE);
	memset(resultBuffer, 0, sizeof(float) * FT_BUFFER_SIZE);
	CudaFT fourier;
	fourier.setDims(FT_BUFFER_SIZE);

	// Play waveform
	while (!IsAudioDeviceReady());
	PlaySound(sound);
	float beginTime = GetTime();

	// Colors for background and bars
	Color bgColor = WHITE, barColor = WHITE;

	while (!WindowShouldClose() && IsSoundPlaying(sound)) {

		float time = GetTime() - beginTime;

		bgColor.r = floorf(127 * (sin(time + 0 * (2 * M_PI / 3)) + 1));
		bgColor.g = floorf(127 * (sin(time + 1 * (2 * M_PI / 3)) + 1));
		bgColor.b = floorf(127 * (sin(time + 2 * (2 * M_PI / 3)) + 1));
		barColor.r = floorf(127 * (cos(time + 0 * (2 * M_PI / 3)) + 1));
		barColor.g = floorf(127 * (cos(time + 1 * (2 * M_PI / 3)) + 1));
		barColor.b = floorf(127 * (cos(time + 2 * (2 * M_PI / 3)) + 1));

		UpdateFourierBuffer(sampleBuffer, samples, FT_BUFFER_SIZE, wave.frameCount, wave.channels, wave.sampleRate, time);
		fourier.transformMag(sampleBuffer, outputBuffer);

		int freq = floorf((GetMousePosition().x / (screenWidth * 2.f / FT_BUFFER_SIZE))) * (wave.sampleRate / FT_BUFFER_SIZE);

		BeginDrawing();
		ClearBackground(bgColor);
		for (int i = 0; i < FT_BUFFER_SIZE / 2; i++) {
			resultBuffer[i] = (SMOOTHING_CONST * resultBuffer[i]) + ((1.f - SMOOTHING_CONST) * outputBuffer[i]);
			Vector2 size = { (float) screenWidth  * 2.f / FT_BUFFER_SIZE, log(resultBuffer[i] * 1000) * 100 };
			Vector2 pos = { i * size.x, screenHeight - size.y };
			DrawRectangleV(pos, size, barColor);
		}
		DrawText(TextFormat("%d Hz", freq), 10, 10, 20, WHITE);
		EndDrawing();
	}

	// Cleanup
	StopSound(sound);
	UnloadSound(sound);
	UnloadWave(wave);
	CloseAudioDevice();
	UnloadWaveSamples(samples);
	CloseWindow();
	return 0;
}
