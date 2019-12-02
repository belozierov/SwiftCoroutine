//
//  URLSession+Extensions.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 28.11.2019.
//  Copyright © 2019 Alex Belozierov. All rights reserved.
//

import Foundation
import SwiftCoroutine

extension URLSession {
    
    typealias DataResponse = (data: Data, response: URLResponse)
    
    func data(for url: URL) -> CoFuture<DataResponse> {
        let promise = CoPromise<DataResponse>()
        dataTask(with: url) {
            if let error = $2 {
                promise.send(error: error)
            } else if let data = $0, let response = $1 {
                promise.send((data, response))
            } else {
                promise.send(error: URLError(.badServerResponse))
            }
        }.resume()
        return promise
    }
    
}
