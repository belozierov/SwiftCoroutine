//
//  CoFutureFlatMapTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureFlatMapTests: XCTestCase {
    
    func testFlatMapSuccess() {
        let promise = CoPromise<Int>()
        let flat = promise.flatMap { value in
            promise.map { value + $0 }
        }.flatMapError { _ in
            XCTFail()
            return CoPromise<Int>()
        }
        promise.success(3)
        XCTAssertEqual(flat.result, 6)
    }
    
    func testFlatMapFail() {
        let future = CoPromise<Int>()
        future.cancel()
        let flat = future.flatMap { value in
            future.map { value + $0 }
        }
        XCTAssertTrue(flat.isCanceled)
        let flat2 = flat.flatMapError { _ in
            CoFuture.init(result: .success(1))
        }
        XCTAssertEqual(flat2.result, 1)
    }
    
    func testFlatMapResult() {
        let promise = CoPromise<Int>()
        let flat = promise.flatMapResult { result in
            CoFuture(result: result.map { $0 + 2 })
        }
        promise.success(3)
        XCTAssertEqual(flat._result, 5)
    }
    
}
