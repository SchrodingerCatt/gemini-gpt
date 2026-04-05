import traceback # ეს აუცილებლად დაამატე ფაილის თავში იმპორტებთან!

@app.post("/process_query")
async def chat_endpoint(
    user_id: str = Form(...),
    prompt: str = Form(...),
    model_choice: str = Form("gemini"),
    image: Optional[UploadFile] = File(None),
    audio: Optional[UploadFile] = File(None)
):
    user_info = get_user_data(user_id)
    
    # [DEBUG] ლოგირება
    if image: print(f"[DEBUG] მოვიდა ფოტო: {image.filename}", flush=True)
    if audio: print(f"[DEBUG] მოვიდა აუდიო: {audio.filename}", flush=True)
    
    has_media = image is not None or audio is not None
    if has_media and user_info["media_count"] >= 1000:
        raise HTTPException(status_code=429, detail="მედია ლიმიტი ამოწურულია.")

    try:
        # Chroma DB-ში ძებნა
        docs = vector_store.similarity_search(prompt, k=2)
        context = "\n".join([d.page_content for d in docs])
        full_query = f"დამხმარე მასალა: {context}\n\nმომხმარებელი: {prompt}"

        if model_choice == "gpt":
            messages = [{"role": "system", "content": SYSTEM_INSTRUCTION}]
            content = [{"type": "text", "text": full_query}]
            if image:
                img_b64 = base64.b64encode(await image.read()).decode('utf-8')
                content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}})
            messages.append({"role": "user", "content": content})
            res = client_openai.chat.completions.create(model="gpt-4o", messages=messages)
            ai_text = res.choices[0].message.content
        else:
            model = genai.GenerativeModel(model_name=MY_GEMINI_MODEL, system_instruction=SYSTEM_INSTRUCTION)
            chat_session = model.start_chat(history=user_info["history"])
            parts = [full_query]
            
            # აუდიოსა და ფოტოს წაკითხვა დამატებამდე
            if image:
                image_data = await image.read()
                parts.append({"mime_type": image.content_type, "data": image_data})
            if audio:
                audio_data = await audio.read()
                parts.append({"mime_type": audio.content_type, "data": audio_data})
                
            response = chat_session.send_message(parts)
            ai_text = response.text

        user_info["history"].append({"role": "user", "parts": [prompt]})
        user_info["history"].append({"role": "model", "parts": [ai_text]})
        if has_media: user_info["media_count"] += 1
        
        return {"response": ai_text, "media_remaining": 1000 - user_info["media_count"]}

    except Exception as e:
        # აი, ეს არის მთავარი დიაგნოსტიკური ნაწილი:
        error_details = traceback.format_exc()
        print(f"[ERROR] მოხდა შეცდომა: {error_details}", flush=True)
        
        # ვაბრუნებთ დეტალურ ინფორმაციას ბრაუზერისთვის
        return JSONResponse(
            status_code=500,
            content={
                "detail": str(e),
                "traceback": error_details,
                "message": "შეცდომა ბექენდში. ნახეთ traceback დეტალებისთვის."
            }
        )
