"use server"

export async function transformImage(imageData: string): Promise<{ transformedImage: string; selectedStyle: string }> {
  try {
    const apiUrl = "http://localhost:8000/transform-image";
    const timeout = 20 * 60 * 1000;

    const controller = new AbortController();
    const timeoutId = setTimeout(() => {
      console.log("Timeout , cancelling request...");
      controller.abort();
    }, timeout);

    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ image_data: imageData }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);
    console.log("Request completed, clearing timeout...");

    if (!response.ok) {
      throw new Error(`Error API : ${response.statusText}`);
    }

    const result = await response.json();
    return {
      transformedImage: result.transformed_image,
      selectedStyle: result.selected_style
    };
  } catch (error: any) {
    if (error.name === "AbortError") {
      console.log("Timeout error: request was aborted");
      throw new Error("Timeout error: request was aborted");
    }
    console.error("Error : ", error);
    throw error;
  }
}
