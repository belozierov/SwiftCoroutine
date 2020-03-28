//
//  CoFutureCombineTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 17.03.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
import Combine
import SwiftCoroutine
import Foundation

@available(OSX 10.15, iOS 13.0, *)
class CoFutureCombineTests: XCTestCase {
    
    func testSubscribe() {
        let exp = expectation(description: "testSubscription")
        let future = Future<Int, Never> { promise in
            DispatchQueue.global().asyncAfter(deadline: .now() + .seconds(1)) {
                promise(.success(1))
            }
        }.delay(for: .seconds(1), scheduler: DispatchQueue.global()).subscribeCoFuture()
        DispatchQueue.global().coroutine {
            XCTAssertEqual(try future.await(), 1)
            exp.fulfill()
        }
        wait(for: [exp], timeout: 3)
    }
    
    func testSubscription() {
        let promise = CoPromise<Int>()
        let publisher = promise.publisher().map { $0 + 1 }
            .sink(receiveCompletion: {
                switch $0 {
                case .finished: break
                case .failure(let error):
                    XCTFail(error.localizedDescription)
                }
            }, receiveValue: {
            XCTAssertEqual($0, 2)
        })
        promise.success(1)
    }
    
}
