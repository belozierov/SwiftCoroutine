//
//  URLSession.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 14.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import Foundation.NSURLSession
import SwiftCoroutine

extension URLSession {
    
    public typealias DataResponse = (data: Data, response: URLResponse)
    
    @inlinable public func dataTaskFuture(for url: URL) -> CoFuture<DataResponse> {
        dataTaskFuture(for: URLRequest(url: url))
    }
    
    public func dataTaskFuture(for urlRequest: URLRequest) -> CoFuture<(data: Data, response: URLResponse)> {
        let promise = CoPromise<DataResponse>()
        let task = dataTask(with: urlRequest) {
            if let error = $2 {
                promise.fail(error)
            } else if let data = $0, let response = $1 {
                promise.success((data, response))
            } else {
                promise.fail(URLError(.badServerResponse))
            }
        }
        task.resume()
        promise.whenCanceled(task.cancel)
        return promise
    }
    
}
