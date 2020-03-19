//
//  CoFutureMapTests.swift
//  SwiftCoroutine
//
//  Created by Alex Belozierov on 03.02.2020.
//  Copyright © 2020 Alex Belozierov. All rights reserved.
//

import XCTest
@testable import SwiftCoroutine

class CoFutureMapTests: XCTestCase {
    
    struct TestError: Error {}
    
    func testMap() {
        let exp = expectation(description: #function)
        exp.expectedFulfillmentCount = 8
        let promise = CoPromise<Int>()
        func test() {
            promise.map { $0 + 1 }.always {
                guard case .success(let result) = $0 else { return }
                XCTAssertEqual(result, 1)
                exp.fulfill()
            }.recover { _ in
                XCTFail()
                return 200
            }.mapResult { _ in
                Result<Int, Error>.failure(TestError())
            }.always { _ in
                exp.fulfill()
            }.always {
                guard case .failure(let error) = $0 else { return }
                XCTAssertTrue(error is TestError)
                exp.fulfill()
            }.recover {
                XCTAssertTrue($0 is TestError)
                return 2
            }.whenSuccess {
                XCTAssertEqual($0, 2)
                exp.fulfill()
            }
        }
        test()
        promise.success(0)
        test()
        wait(for: [exp], timeout: 1)
    }
    
}
